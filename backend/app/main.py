from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import json
import os
import sys
from datetime import datetime
import joblib
import shap
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
from agent.evidence_agent import EvidenceAgent

try:
    from app.db import init_db, SessionLocal
    from app.models import Transaction, Decision, AuditLog
except ImportError:
    from db import init_db, SessionLocal
    from models import Transaction, Decision, AuditLog

app = FastAPI(title="Chargeback Evidence Responder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve absolute paths for model and metrics files sitting next to main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chargeback_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

# Load performance metrics from metrics.json
if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}")
with open(METRICS_PATH, "r") as f:
    metrics_data = json.load(f)

FEATURE_ORDER = metrics_data.get("features", [
    "Amount",
    "tx_count_24h",
    "minutes_since_last_tx",
    "amount_vs_card_avg",
    "is_odd_hour"
])
THRESHOLD = metrics_data.get("threshold", 0.8553)

# Load LightGBM model and initialize explainer
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)
evidence_agent = EvidenceAgent()

@app.on_event("startup")
def startup():
    try:
        init_db()   # creates the 3 tables in your Neon DB the first time this runs
    except Exception as e:
        print(f"Warning: Could not initialize database at startup: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ScoreRequest(BaseModel):
    card_number: str
    amount: float
    tx_count_24h: int
    minutes_since_last_tx: float
    amount_vs_card_avg: float
    transaction_time: datetime

class EvidenceRequest(BaseModel):
    evidence_packet: str

@app.post("/score")
def score_transaction(req: ScoreRequest, db: Session = Depends(get_db)):
    is_odd_hour = req.transaction_time.hour < 6 or req.transaction_time.hour >= 22

    # 1. Log transaction in the transactions table
    txn = Transaction(
        card_number=req.card_number,
        amount=req.amount,
        tx_count_24h=req.tx_count_24h,
        minutes_since_last_tx=req.minutes_since_last_tx,
        amount_vs_card_avg=req.amount_vs_card_avg,
        is_odd_hour=is_odd_hour,
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)

    # 2. Prepare the named dataframe for LightGBM.
    # Note: 'Amount' is capitalized as expected by the trained model features.
    input_data = pd.DataFrame([{
        'Amount': req.amount,
        'tx_count_24h': req.tx_count_24h,
        'minutes_since_last_tx': req.minutes_since_last_tx,
        'amount_vs_card_avg': req.amount_vs_card_avg,
        'is_odd_hour': int(is_odd_hour)
    }], columns=FEATURE_ORDER)

    # 3. Model prediction
    probabilities = model.predict_proba(input_data)
    score = float(probabilities[0, 1])
    decision_label = "fight" if score >= THRESHOLD else "auto_refund"

    # 4. SHAP-based feature importance/reasoning
    sv = explainer.shap_values(input_data)
    # Handle SHAP output list vs ndarray variations
    row_shap = sv[1][0] if isinstance(sv, list) else sv[0]
    
    reasons = sorted(
        [(name, float(val)) for name, val in zip(FEATURE_ORDER, row_shap)],
        key=lambda x: -abs(x[1])
    )

    # 5. Log decision in the decisions table
    decision = Decision(
        transaction_id=txn.id,
        score=score,
        decision=decision_label,
        threshold_used=THRESHOLD,
        top_reasons=json.dumps(reasons)
    )
    db.add(decision)
    db.commit()
    db.refresh(decision)

    # 6. Log audit event "scored"
    log = AuditLog(
        decision_id=decision.id,
        event="scored",
        detail=f"score={score:.4f}, decision={decision_label}"
    )
    db.add(log)
    db.commit()

    return {
        "transaction_id": txn.id,
        "score": score,
        "decision": decision_label,
        "top_reasons": reasons,
        "decision_id": decision.id
    }

@app.get("/transactions")
def get_transactions(limit: int = Query(20, ge=1), db: Session = Depends(get_db)):
    results = db.query(
        Transaction.id.label("transaction_id"),
        Transaction.card_number,
        Transaction.amount,
        Decision.id.label("decision_id"),
        Decision.score,
        Decision.decision,
        Transaction.created_at
    ).join(Decision, Transaction.id == Decision.transaction_id).order_by(Transaction.created_at.desc()).limit(limit).all()
    
    return [
        {
            "transaction_id": r.transaction_id,
            "card_number": r.card_number,
            "amount": r.amount,
            "decision_id": r.decision_id,
            "score": r.score,
            "decision": r.decision,
            "created_at": str(r.created_at)
        }
        for r in results
    ]

@app.get("/decisions/{decision_id}")
def get_decision(decision_id: int, db: Session = Depends(get_db)):
    decision = db.query(Decision).filter(Decision.id == decision_id).first()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    txn = db.query(Transaction).filter(Transaction.id == decision.transaction_id).first()
    
    txn_dict = None
    if txn:
        txn_dict = {
            "id": txn.id,
            "card_number": txn.card_number,
            "amount": txn.amount,
            "created_at": str(txn.created_at)
        }
        
    return {
        "id": decision.id,
        "decision_id": decision.id,
        "transaction_id": decision.transaction_id,
        "score": decision.score,
        "decision": decision.decision,
        "threshold_used": decision.threshold_used,
        "top_reasons": json.loads(decision.top_reasons),
        "evidence_packet": decision.evidence_packet,
        "created_at": str(decision.created_at),
        "transaction": txn_dict
    }

@app.post("/evidence/{decision_id}")
def post_evidence(decision_id: int, req: EvidenceRequest, db: Session = Depends(get_db)):
    decision = db.query(Decision).filter(Decision.id == decision_id).first()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    if decision.decision != "fight":
        raise HTTPException(status_code=400, detail="Evidence generation is only allowed for 'fight' decisions")
    
    # Save evidence packet to decision itself
    decision.evidence_packet = req.evidence_packet
    db.commit()
    
    # Log evidence generation event
    log = AuditLog(
        decision_id=decision.id,
        event="evidence_generated",
        detail=req.evidence_packet
    )
    db.add(log)
    db.commit()
    
    return {
        "status": "success",
        "message": "Evidence generated successfully",
        "decision_id": decision_id
    }

@app.post("/agent/generate-evidence/{decision_id}")
def generate_evidence_with_agent(decision_id: int, db: Session = Depends(get_db)):
    """
    Generate evidence using the LangGraph evidence agent.
    This endpoint turns a 'fight' decision into an actual dispute-evidence draft.
    """
    decision = db.query(Decision).filter(Decision.id == decision_id).first()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    if decision.decision != "fight":
        raise HTTPException(status_code=400, detail="Evidence generation is only allowed for 'fight' decisions")

    try:
        # Run the LangGraph agent with the decision's data
        top_reasons = json.loads(decision.top_reasons)
        result = evidence_agent.generate_evidence(
            decision_id=decision.id,
            top_reasons=top_reasons,
        )
        
        # Store the resulting final_evidence onto decision.evidence_packet
        decision.evidence_packet = result["final_evidence"]
        db.commit()
        
        # Log "evidence_generated" event to the audit trail
        log = AuditLog(
            decision_id=decision.id,
            event="evidence_generated",
            detail=f"agent_generated={result['is_valid']}, graceful_decline={result['graceful_decline']}"
        )
        db.add(log)
        db.commit()
        
        return {
            "final_evidence": result["final_evidence"],
            "is_valid": result["is_valid"],
            "graceful_decline": result["graceful_decline"],
            "decision_id": decision_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evidence generation failed: {str(e)}")


@app.post("/evidence/{decision_id}/generate")
def generate_evidence(decision_id: int, db: Session = Depends(get_db)):
    """Legacy endpoint - redirects to the new agent endpoint."""
    return generate_evidence_with_agent(decision_id, db)

@app.get("/audit/{decision_id}")
def get_audit(decision_id: int, db: Session = Depends(get_db)):
    logs = db.query(AuditLog).filter(AuditLog.decision_id == decision_id).order_by(AuditLog.created_at.asc()).all()
    return [{"event": l.event, "detail": l.detail, "at": str(l.created_at)} for l in logs]

@app.get("/metrics")
def get_metrics():
    return metrics_data
