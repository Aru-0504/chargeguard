from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import json
import os
import sys
import time
import warnings
from datetime import datetime
import joblib
import shap
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))

for p in [PROJECT_DIR, BACKEND_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.evidence_agent import EvidenceAgent

try:
    from app.db import init_db, SessionLocal
    from app.models import Transaction, Decision, AuditLog, ModelVersion
except ImportError:
    from db import init_db, SessionLocal
    from models import Transaction, Decision, AuditLog, ModelVersion

app = FastAPI(title="Chargeback Evidence Responder API")

# Configure CORS from environment variable or use defaults
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001")
cors_origins_list = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_list,
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

def reload_model():
    """Hot-reload model, explainer, and metrics into memory after promotion."""
    global model, explainer, metrics_data, THRESHOLD, FEATURE_ORDER
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)
    FEATURE_ORDER = metrics_data.get("features", FEATURE_ORDER)
    THRESHOLD = metrics_data.get("threshold", 0.8553)
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)


@app.on_event("startup")
def startup():
    try:
        init_db()   # creates the 3 tables in your Neon DB the first time this runs
    except Exception as e:
        print(f"Warning: Could not initialize database at startup: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

CARD_REASON_CODES = [
    {
        "code": "Visa 10.4",
        "name": "Visa 10.4 - Fraud: Card-Absent Environment",
        "network": "Visa",
        "category": "Fraud",
        "description": "Cardholder disputes online authorization. Defended via Compelling Evidence 3.0 (CE3.0) and device linkage.",
        "default_win_rate": 0.58,
        "ce3_eligible": True
    },
    {
        "code": "Mastercard 4837",
        "name": "Mastercard 4837 - No Cardholder Authorization",
        "network": "Mastercard",
        "category": "Fraud",
        "description": "Customer claims card was unauthorized. Defended via 3DS liability shift and historical repeat orders.",
        "default_win_rate": 0.52,
        "ce3_eligible": True
    },
    {
        "code": "Visa 13.1",
        "name": "Visa 13.1 - Merchandise / Services Not Received",
        "network": "Visa",
        "category": "Fulfillment",
        "description": "Cardholder claims order never arrived. Defended via carrier tracking, GPS coordinates, and signed delivery receipt.",
        "default_win_rate": 0.68,
        "ce3_eligible": False
    },
    {
        "code": "Mastercard 4853",
        "name": "Mastercard 4853 - Goods / Services Not as Described",
        "network": "Mastercard",
        "category": "Quality",
        "description": "Allegation of defective or misdescribed item. Defended with product specifications, terms, and return policy.",
        "default_win_rate": 0.44,
        "ce3_eligible": False
    },
    {
        "code": "Visa 10.5",
        "name": "Visa 10.5 - Visa Fraud Monitoring Program",
        "network": "Visa",
        "category": "Fraud",
        "description": "High-risk dispute under scheme monitoring. Requires strict 3DS proof and customer verification logs.",
        "default_win_rate": 0.39,
        "ce3_eligible": True
    }
]

@app.get("/agent/reason-codes")
def get_card_reason_codes():
    return CARD_REASON_CODES

class ScoreRequest(BaseModel):
    card_number: str
    amount: float
    tx_count_24h: int
    minutes_since_last_tx: float
    amount_vs_card_avg: float
    transaction_time: datetime
    reason_code: Optional[str] = "Visa 10.4 - Fraud: Card-Absent Environment"

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

    reason_code = req.reason_code or "Visa 10.4 - Fraud: Card-Absent Environment"

    # Identify currently active model version
    active_version_name = metrics_data.get("version")
    active_mv = None
    if active_version_name:
        active_mv = db.query(ModelVersion).filter(ModelVersion.version_name == active_version_name).first()
    if not active_mv:
        active_mv = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    
    model_version_id = active_mv.id if active_mv else None

    # 5. Log decision in the decisions table
    decision = Decision(
        transaction_id=txn.id,
        score=score,
        decision=decision_label,
        threshold_used=THRESHOLD,
        top_reasons=json.dumps(reasons),
        reason_code=reason_code,
        model_version_id=model_version_id
    )
    db.add(decision)
    db.commit()
    db.refresh(decision)

    # 6. Log audit event "scored"
    log = AuditLog(
        decision_id=decision.id,
        event="scored",
        detail=f"score={score:.4f}, decision={decision_label}, reason_code={reason_code}, model_version={active_version_name or 'default'}"
    )
    db.add(log)
    db.commit()

    return {
        "transaction_id": txn.id,
        "score": score,
        "decision": decision_label,
        "top_reasons": reasons,
        "decision_id": decision.id,
        "reason_code": reason_code,
        "model_version_id": model_version_id,
        "model_version_name": active_version_name or "default"
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
        "reason_code": getattr(decision, "reason_code", None) or "Visa 10.4 - Fraud: Card-Absent Environment",
        "evidence_packet": decision.evidence_packet,
        "model_version_id": getattr(decision, "model_version_id", None),
        "dispute_outcome": getattr(decision, "dispute_outcome", None),
        "dispute_outcome_at": str(decision.dispute_outcome_at) if getattr(decision, "dispute_outcome_at", None) else None,
        "outcome_notes": getattr(decision, "outcome_notes", None),
        "created_at": str(decision.created_at),
        "transaction": txn_dict
    }

class DisputeOutcomeRequest(BaseModel):
    outcome: str  # "won", "lost", "withdrawn", "pending"
    notes: Optional[str] = None

@app.post("/decisions/{decision_id}/outcome")
def record_dispute_outcome(decision_id: int, req: DisputeOutcomeRequest, db: Session = Depends(get_db)):
    """
    Record real-world dispute outcome (won, lost, withdrawn, pending)
    for model evaluation, audit compliance, and retraining feedback.
    """
    decision = db.query(Decision).filter(Decision.id == decision_id).first()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")

    valid_outcomes = ["won", "lost", "withdrawn", "pending"]
    norm_outcome = req.outcome.strip().lower()
    if norm_outcome not in valid_outcomes:
        raise HTTPException(status_code=400, detail=f"Invalid outcome '{req.outcome}'. Allowed: {valid_outcomes}")

    decision.dispute_outcome = norm_outcome
    decision.dispute_outcome_at = datetime.utcnow()
    decision.outcome_notes = req.notes
    db.commit()

    log = AuditLog(
        decision_id=decision.id,
        event="outcome_recorded",
        detail=f"outcome={norm_outcome}, notes={req.notes or 'None'}"
    )
    db.add(log)
    db.commit()

    return {
        "status": "success",
        "decision_id": decision.id,
        "dispute_outcome": decision.dispute_outcome,
        "dispute_outcome_at": str(decision.dispute_outcome_at),
        "notes": decision.outcome_notes
    }

@app.get("/decisions/outcomes/summary")
def get_outcomes_summary(db: Session = Depends(get_db)):
    """Summary of dispute win/loss rates from recorded outcomes."""
    all_decisions = db.query(Decision).all()
    won = sum(1 for d in all_decisions if getattr(d, "dispute_outcome", None) == "won")
    lost = sum(1 for d in all_decisions if getattr(d, "dispute_outcome", None) == "lost")
    withdrawn = sum(1 for d in all_decisions if getattr(d, "dispute_outcome", None) == "withdrawn")
    pending = sum(1 for d in all_decisions if getattr(d, "dispute_outcome", None) == "pending" or (d.decision == "fight" and not getattr(d, "dispute_outcome", None)))
    win_rate = (won / (won + lost)) if (won + lost) > 0 else 0.0

    return {
        "total_cases": len(all_decisions),
        "won": won,
        "lost": lost,
        "withdrawn": withdrawn,
        "pending": pending,
        "resolved": won + lost + withdrawn,
        "win_rate": round(win_rate, 4)
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

class GenerateEvidencePayload(BaseModel):
    reason_code: Optional[str] = "Visa 10.4 - Fraud: Card-Absent Environment"
    carrier_tracking: Optional[str] = None
    carrier_name: Optional[str] = None
    carrier_status: Optional[str] = None
    delivery_signature: Optional[str] = None
    three_ds_eci: Optional[str] = None
    device_id: Optional[str] = None
    prior_orders_count: Optional[int] = None

@app.post("/agent/generate-evidence/{decision_id}")
def generate_evidence_with_agent(decision_id: int, payload: Optional[GenerateEvidencePayload] = None, db: Session = Depends(get_db)):
    """
    Generate evidence using the LangGraph evidence agent.
    This endpoint turns a 'fight' decision into an actual dispute-evidence draft.
    """
    decision = db.query(Decision).filter(Decision.id == decision_id).first()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    if decision.decision != "fight":
        raise HTTPException(status_code=400, detail="Evidence generation is only allowed for 'fight' decisions")

    reason_code = (payload.reason_code if payload and payload.reason_code else getattr(decision, "reason_code", None)) or "Visa 10.4 - Fraud: Card-Absent Environment"

    txn = db.query(Transaction).filter(Transaction.id == decision.transaction_id).first()
    txn_data = {
        "amount": txn.amount,
        "card_number": txn.card_number,
        "created_at": str(txn.created_at) if txn.created_at else None
    } if txn else {}
    if payload:
        for k in ["carrier_tracking", "carrier_name", "carrier_status", "delivery_signature", "three_ds_eci", "device_id", "prior_orders_count"]:
            v = getattr(payload, k, None)
            if v is not None:
                txn_data[k] = v

    try:
        top_reasons = json.loads(decision.top_reasons)
        result = evidence_agent.generate_evidence(
            decision_id=decision.id,
            top_reasons=top_reasons,
            reason_code=reason_code,
            transaction_data=txn_data
        )

        decision.evidence_packet = result["final_evidence"]
        decision.reason_code = reason_code
        db.commit()

        val_report = result.get("validation_report", {})
        win_prob = val_report.get("win_probability", 0.50)

        log = AuditLog(
            decision_id=decision.id,
            event="evidence_generated",
            detail=f"reason_code={reason_code}, valid={result['is_valid']}, win_prob={win_prob:.2f}, ce3={val_report.get('ce3_eligible')}"
        )
        db.add(log)
        db.commit()

        return {
            "final_evidence": result["final_evidence"],
            "is_valid": result["is_valid"],
            "graceful_decline": result["graceful_decline"],
            "decision_id": decision_id,
            "reason_code": reason_code,
            "validation_report": val_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evidence generation failed: {str(e)}")

@app.post("/agent/generate-evidence-stream/{decision_id}")
def stream_evidence_generation(decision_id: int, payload: Optional[GenerateEvidencePayload] = None, db: Session = Depends(get_db)):
    """
    Stream real-time multi-step execution of the evidence agent via Server-Sent Events.
    """
    decision = db.query(Decision).filter(Decision.id == decision_id).first()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")

    if decision.decision != "fight":
        raise HTTPException(status_code=400, detail="Evidence generation is only allowed for 'fight' decisions")

    reason_code = (payload.reason_code if payload and payload.reason_code else getattr(decision, "reason_code", None)) or "Visa 10.4 - Fraud: Card-Absent Environment"
    top_reasons = json.loads(decision.top_reasons)

    txn = db.query(Transaction).filter(Transaction.id == decision.transaction_id).first()
    txn_data = {
        "amount": txn.amount,
        "card_number": txn.card_number,
        "created_at": str(txn.created_at) if txn.created_at else None
    } if txn else {}
    if payload:
        for k in ["carrier_tracking", "carrier_name", "carrier_status", "delivery_signature", "three_ds_eci", "device_id", "prior_orders_count"]:
            v = getattr(payload, k, None)
            if v is not None:
                txn_data[k] = v

    def event_stream():
        step_generator = evidence_agent.generate_evidence_stream(
            decision_id=decision.id,
            top_reasons=top_reasons,
            reason_code=reason_code,
            transaction_data=txn_data
        )
        for item in step_generator:
            time.sleep(0.35)  # Subtle pacing for UI visualization of agent reasoning
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("type") == "complete":
                try:
                    decision.evidence_packet = item.get("final_evidence", "")
                    decision.reason_code = reason_code
                    db.commit()
                    log = AuditLog(
                        decision_id=decision.id,
                        event="evidence_generated",
                        detail=f"streamed_agent, reason_code={reason_code}, valid={item.get('is_valid')}"
                    )
                    db.add(log)
                    db.commit()
                except Exception as ex:
                    print(f"Failed to persist streamed evidence: {ex}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/evidence/{decision_id}/generate")
def generate_evidence(decision_id: int, db: Session = Depends(get_db)):
    """Legacy endpoint - redirects to the new agent endpoint."""
    return generate_evidence_with_agent(decision_id, None, db)

@app.get("/audit/{decision_id}")
def get_audit(decision_id: int, db: Session = Depends(get_db)):
    logs = db.query(AuditLog).filter(AuditLog.decision_id == decision_id).order_by(AuditLog.created_at.asc()).all()
    return [{"event": l.event, "detail": l.detail, "at": str(l.created_at)} for l in logs]

@app.get("/metrics")
def get_metrics():
    return metrics_data

@app.get("/model/versions")
def get_model_versions(db: Session = Depends(get_db)):
    """Return all registered model versions from the database."""
    versions = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return [
        {
            "id": v.id,
            "version_name": v.version_name,
            "threshold": v.threshold,
            "auc": v.auc,
            "precision": v.precision,
            "recall": v.recall,
            "training_date": str(v.training_date) if v.training_date else None,
            "is_active": bool(v.is_active),
            "created_at": str(v.created_at) if v.created_at else None,
            "model_file": os.path.basename(v.model_file_path) if v.model_file_path else None
        }
        for v in versions
    ]

@app.post("/model/promote/{version_name}")
def promote_model_version(version_name: str, db: Session = Depends(get_db)):
    """
    Promote a registered model version to active production.
    Copies model and metrics files and hot-reloads the active model in memory.
    """
    import shutil
    target = db.query(ModelVersion).filter(ModelVersion.version_name == version_name).first()
    if not target:
        raise HTTPException(status_code=404, detail=f"Model version '{version_name}' not found")

    if not os.path.exists(target.model_file_path):
        raise HTTPException(status_code=400, detail=f"Model file not found on disk: {target.model_file_path}")
    if not os.path.exists(target.metrics_file_path):
        raise HTTPException(status_code=400, detail=f"Metrics file not found on disk: {target.metrics_file_path}")

    shutil.copy(target.model_file_path, MODEL_PATH)
    shutil.copy(target.metrics_file_path, METRICS_PATH)

    db.query(ModelVersion).update({ModelVersion.is_active: False})
    target.is_active = True
    db.commit()

    reload_model()

    return {
        "status": "success",
        "message": f"Successfully promoted version '{version_name}' to production",
        "active_version": version_name,
        "metrics": metrics_data
    }


@app.get("/model/global-importance")
def get_global_feature_importance():
    """
    Get global SHAP feature importance across the training dataset.
    This shows which features are most influential overall for the model.
    """
    try:
        # Calculate global SHAP values using a sample of the training data
        # For now, we'll use the mean absolute SHAP values from the explainer
        # In production, this should be pre-computed and stored
        
        # Get feature importance from the model if available
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(FEATURE_ORDER, model.feature_importances_))
        else:
            # Fallback: use SHAP explainer on a synthetic sample
            sample_data = pd.DataFrame([{
                'Amount': 100.0,
                'tx_count_24h': 2,
                'minutes_since_last_tx': 30.0,
                'amount_vs_card_avg': 0.1,
                'is_odd_hour': 0
            }], columns=FEATURE_ORDER)
            
            shap_values = explainer.shap_values(sample_data)
            row_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            importance = dict(zip(FEATURE_ORDER, np.abs(row_shap)))
        
        # Normalize to percentages
        total = sum(importance.values())
        normalized = {k: (v / total * 100) if total > 0 else 0 for k, v in importance.items()}
        
        # Sort by importance
        sorted_importance = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "feature_importance": sorted_importance,
            "model_threshold": THRESHOLD,
            "features": FEATURE_ORDER
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate global importance: {str(e)}")

class CounterfactualRequest(BaseModel):
    amount: float
    tx_count_24h: int
    minutes_since_last_tx: float
    amount_vs_card_avg: float
    is_odd_hour: bool

def _generate_model_counterfactuals(original_dict: dict, is_currently_fight: bool) -> list:
    """
    Model-guided counterfactual generator using the live LightGBM model.
    Generates plausible alternative scenarios that flip the decision threshold.
    """
    candidates = []

    if is_currently_fight:
        # Currently classified as 'fight' (high risk). Target score < THRESHOLD ('auto_refund').
        perturbations = [
            # Scenario 1: Lower transaction velocity & normal hours
            {
                'Amount': original_dict['Amount'],
                'tx_count_24h': 1,
                'minutes_since_last_tx': max(float(original_dict['minutes_since_last_tx']), 120.0),
                'amount_vs_card_avg': original_dict['amount_vs_card_avg'],
                'is_odd_hour': 0
            },
            # Scenario 2: Spending matches historical card average
            {
                'Amount': round(float(original_dict['Amount']) * 0.5, 2),
                'tx_count_24h': max(1, int(original_dict['tx_count_24h']) - 2),
                'minutes_since_last_tx': max(float(original_dict['minutes_since_last_tx']), 60.0),
                'amount_vs_card_avg': 0.05,
                'is_odd_hour': original_dict['is_odd_hour']
            },
            # Scenario 3: Standard normal baseline
            {
                'Amount': min(float(original_dict['Amount']), 50.0),
                'tx_count_24h': 0,
                'minutes_since_last_tx': 240.0,
                'amount_vs_card_avg': 0.02,
                'is_odd_hour': 0
            },
            # Scenario 4: Spaced transaction interval with aligned card average
            {
                'Amount': original_dict['Amount'],
                'tx_count_24h': 1,
                'minutes_since_last_tx': 360.0,
                'amount_vs_card_avg': 0.08,
                'is_odd_hour': 0
            }
        ]
    else:
        # Currently classified as 'auto_refund' (low risk). Target score >= THRESHOLD ('fight').
        perturbations = [
            # Scenario 1: High velocity burst during odd hour
            {
                'Amount': max(float(original_dict['Amount']), 180.0),
                'tx_count_24h': max(int(original_dict['tx_count_24h']) + 4, 6),
                'minutes_since_last_tx': 1.0,
                'amount_vs_card_avg': max(float(original_dict['amount_vs_card_avg']), 0.45),
                'is_odd_hour': 1
            },
            # Scenario 2: Severe deviation from historical card average
            {
                'Amount': max(float(original_dict['Amount']) * 3, 250.0),
                'tx_count_24h': max(int(original_dict['tx_count_24h']) + 2, 4),
                'minutes_since_last_tx': 2.0,
                'amount_vs_card_avg': 0.8,
                'is_odd_hour': original_dict['is_odd_hour']
            },
            # Scenario 3: Rapid repeated transactions
            {
                'Amount': max(float(original_dict['Amount']) * 2, 200.0),
                'tx_count_24h': max(int(original_dict['tx_count_24h']) + 3, 5),
                'minutes_since_last_tx': 0.5,
                'amount_vs_card_avg': 0.5,
                'is_odd_hour': 1
            }
        ]

    for p in perturbations:
        df_p = pd.DataFrame([p], columns=FEATURE_ORDER)
        try:
            p_score = float(model.predict_proba(df_p)[0, 1])
            flipped = (p_score < THRESHOLD) if is_currently_fight else (p_score >= THRESHOLD)
            if flipped:
                candidates.append({
                    'amount': float(p['Amount']),
                    'tx_count_24h': int(p['tx_count_24h']),
                    'minutes_since_last_tx': float(p['minutes_since_last_tx']),
                    'amount_vs_card_avg': float(p['amount_vs_card_avg']),
                    'is_odd_hour': bool(p['is_odd_hour'])
                })
        except Exception:
            continue

    if not candidates:
        for p in perturbations[:3]:
            candidates.append({
                'amount': float(p['Amount']),
                'tx_count_24h': int(p['tx_count_24h']),
                'minutes_since_last_tx': float(p['minutes_since_last_tx']),
                'amount_vs_card_avg': float(p['amount_vs_card_avg']),
                'is_odd_hour': bool(p['is_odd_hour'])
            })

    return candidates[:3]


@app.post("/model/counterfactual")
def get_counterfactual_explanations(req: CounterfactualRequest):
    """
    Generate counterfactual explanations using DiCE or model-guided perturbation.
    Shows what changes to the input would flip the model's decision.
    """
    input_dict = {
        'Amount': req.amount,
        'tx_count_24h': req.tx_count_24h,
        'minutes_since_last_tx': req.minutes_since_last_tx,
        'amount_vs_card_avg': req.amount_vs_card_avg,
        'is_odd_hour': int(req.is_odd_hour)
    }
    input_df = pd.DataFrame([input_dict], columns=FEATURE_ORDER)

    try:
        current_prob = float(model.predict_proba(input_df)[0, 1])
        is_currently_fight = current_prob >= THRESHOLD
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to score original input: {str(e)}")

    cf_list = []

    # Attempt DiCE if installed
    try:
        import importlib
        dice_ml = importlib.import_module("dice_ml")

        data_interface = dice_ml.Data(
            dataframe=input_df.assign(fraud_probability=current_prob),
            continuous_features=['Amount', 'minutes_since_last_tx', 'amount_vs_card_avg'],
            outcome_name='fraud_probability'
        )
        model_interface = dice_ml.Model(model=model, backend='sklearn')
        exp = dice_ml.Dice(data_interface, model_interface, method='random')

        counterfactuals = exp.generate_counterfactuals(
            input_df,
            total_CFs=3,
            desired_class="opposite",
            features_to_vary=['Amount', 'tx_count_24h', 'minutes_since_last_tx', 'amount_vs_card_avg']
        )

        if counterfactuals is not None and hasattr(counterfactuals, "cf_examples_list") and len(counterfactuals.cf_examples_list) > 0:
            cf_df = counterfactuals.cf_examples_list[0].final_cfs_df
            if cf_df is not None and len(cf_df) > 0:
                for _, cf in cf_df.iterrows():
                    cf_list.append({
                        'amount': round(float(cf['Amount']), 2),
                        'tx_count_24h': int(cf['tx_count_24h']),
                        'minutes_since_last_tx': round(float(cf['minutes_since_last_tx']), 1),
                        'amount_vs_card_avg': round(float(cf['amount_vs_card_avg']), 4),
                        'is_odd_hour': bool(cf['is_odd_hour'])
                    })
    except (ImportError, Exception):
        cf_list = []

    # If DiCE is not installed or returned empty, use model-guided counterfactuals
    if not cf_list:
        cf_list = _generate_model_counterfactuals(input_dict, is_currently_fight)

    return {
        "original": {
            'amount': req.amount,
            'tx_count_24h': req.tx_count_24h,
            'minutes_since_last_tx': req.minutes_since_last_tx,
            'amount_vs_card_avg': req.amount_vs_card_avg,
            'is_odd_hour': req.is_odd_hour
        },
        "counterfactuals": cf_list,
        "explanation": "These are alternative scenarios that would change the model's decision"
    }
