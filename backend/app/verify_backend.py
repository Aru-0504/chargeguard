import sys
import os

# Set SQLite fallback for testing before any db imports
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

# Ensure the parent directory is in sys.path so we can import app modules if run from app/ or backend/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("--- Step 1: Testing Imports ---")
try:
    import fastapi
    import uvicorn
    import sqlalchemy
    import psycopg2
    import joblib
    import shap
    import pandas as pd
    import numpy as np
    import dotenv
    import lightgbm
    import httpx
    print("Success: All imports completed successfully!")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

print("\n--- Step 2: Testing Database Connection ---")
try:
    from db import engine, init_db
    from models import Transaction, Decision, AuditLog
    
    # Initialize DB (creates tables if they do not exist)
    print("Initializing database and creating tables if they do not exist...")
    init_db()
    
    # Test connection
    with engine.connect() as conn:
        print(f"Success: Successfully connected to SQLite database for testing!")
except Exception as e:
    print(f"Database Connection Error: {e}")
    print("Please double check DATABASE_URL in backend/.env")
    sys.exit(1)

print("\n--- Step 3: Running FastAPI TestClient End-to-End Suite ---")
try:
    from fastapi.testclient import TestClient
    from main import app
    
    client = TestClient(app)
    
    # 1. POST /score
    print("\n1. Testing POST /score...")
    score_payload = {
      "card_number": "400217******1353",
      "amount": 172.5,
      "tx_count_24h": 5,
      "minutes_since_last_tx": 1.5,
      "amount_vs_card_avg": 0.4,
    "transaction_time": "2026-08-25T02:00:00Z"
    }
    response = client.post("/score", json=score_payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    res_data = response.json()
    decision_id = res_data["decision_id"]
    print(f"Decision ID: {decision_id}, Decision: {res_data['decision']}")
    assert res_data["decision"] == "fight"
    
    # 2. GET /transactions
    print("\n2. Testing GET /transactions...")
    response = client.get("/transactions")
    print(f"Status Code: {response.status_code}")
    txns = response.json()
    print(f"Total Transactions: {len(txns)}")
    assert response.status_code == 200
    assert len(txns) > 0
    # Confirm the card number matches
    assert txns[0]["card_number"] == score_payload["card_number"]
    
    # 3. GET /decisions/{decision_id}
    print(f"\n3. Testing GET /decisions/{decision_id}...")
    response = client.get(f"/decisions/{decision_id}")
    print(f"Status Code: {response.status_code}")
    dec_data = response.json()
    print(f"Decision Data: {dec_data}")
    assert response.status_code == 200
    assert dec_data["id"] == decision_id
    assert "top_reasons" in dec_data
    print("SHAP reasons loaded successfully.")

    # 4. POST /evidence/{decision_id}
    print(f"\n4. Testing POST /evidence/{decision_id}...")
    evidence_payload = {"evidence_packet": "test evidence text"}
    response = client.post(f"/evidence/{decision_id}", json=evidence_payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # 5. GET /audit/{decision_id}
    print(f"\n5. Testing GET /audit/{decision_id}...")
    response = client.get(f"/audit/{decision_id}")
    print(f"Status Code: {response.status_code}")
    audit_logs = response.json()
    print(f"Audit Logs: {audit_logs}")
    assert response.status_code == 200
    assert len(audit_logs) >= 2
    events = [log["event"] for log in audit_logs]
    assert "scored" in events
    assert "evidence_generated" in events
    print("Audit log verification successful.")

    # 6. GET /metrics
    print("\n6. Testing GET /metrics...")
    response = client.get("/metrics")
    print(f"Status Code: {response.status_code}")
    metrics = response.json()
    print(f"Metrics: {metrics}")
    assert response.status_code == 200
    assert "auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics

    # 7. GET /model/global-importance
    print("\n7. Testing GET /model/global-importance...")
    response = client.get("/model/global-importance")
    print(f"Status Code: {response.status_code}")
    importance = response.json()
    print(f"Global Importance: {importance}")
    assert response.status_code == 200
    assert "feature_importance" in importance
    assert "model_threshold" in importance

    # 8. GET /agent/reason-codes
    print("\n8. Testing GET /agent/reason-codes...")
    response = client.get("/agent/reason-codes")
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    rc_list = response.json()
    assert len(rc_list) >= 4
    print(f"Loaded {len(rc_list)} Card Scheme Reason Codes successfully!")

    # 9. POST /agent/generate-evidence/{decision_id}
    print("\n9. Testing POST /agent/generate-evidence/{decision_id}...")
    response = client.post(f"/agent/generate-evidence/{decision_id}", json={"reason_code": "Visa 10.4 - Fraud: Card-Absent Environment"})
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    agent_result = response.json()
    assert "final_evidence" in agent_result
    assert "reason_code" in agent_result
    print("Evidence generated successfully!")

    # 10. POST /agent/generate-evidence-stream/{decision_id}
    print("\n10. Testing POST /agent/generate-evidence-stream/{decision_id}...")
    response = client.post(f"/agent/generate-evidence-stream/{decision_id}", json={"reason_code": "Visa 10.4 - Fraud: Card-Absent Environment"})
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    print("Streaming endpoint returned event-stream successfully!")

    # 11. POST /model/counterfactual
    print("\n11. Testing POST /model/counterfactual...")
    counterfactual_payload = {
        "amount": 172.5,
        "tx_count_24h": 5,
        "minutes_since_last_tx": 1.5,
        "amount_vs_card_avg": 0.4,
        "is_odd_hour": True
    }
    response = client.post("/model/counterfactual", json=counterfactual_payload)
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    cf_result = response.json()
    print(f"Counterfactual Result: {cf_result}")
    assert "original" in cf_result
    assert "counterfactuals" in cf_result
    assert len(cf_result["counterfactuals"]) > 0
    print(f"Generated {len(cf_result['counterfactuals'])} counterfactual scenarios!")

    # 12. Seed and test GET /model/versions
    print("\n12. Testing GET /model/versions...")
    from db import SessionLocal
    from models import ModelVersion
    test_db = SessionLocal()
    candidate_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chargeback_model_v20260929_164908.pkl")
    candidate_metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_v20260929_164908.json")
    if os.path.exists(candidate_model_path) and os.path.exists(candidate_metrics_path):
        existing = test_db.query(ModelVersion).filter(ModelVersion.version_name == "v20260929_164908").first()
        if not existing:
            sample_mv = ModelVersion(
                version_name="v20260929_164908",
                model_file_path=candidate_model_path,
                metrics_file_path=candidate_metrics_path,
                features='["Amount", "tx_count_24h", "minutes_since_last_tx", "amount_vs_card_avg", "is_odd_hour"]',
                threshold=0.8553,
                auc=0.9035,
                precision=0.7053,
                recall=0.5877,
                is_active=False
            )
            test_db.add(sample_mv)
            test_db.commit()
    test_db.close()

    response = client.get("/model/versions")
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    versions = response.json()
    print(f"Registered model versions: {len(versions)}")

    # 13. Test Model Promotion if any version exists
    if versions:
        test_version = versions[0]["version_name"]
        print(f"\n13. Testing POST /model/promote/{test_version}...")
        response = client.post(f"/model/promote/{test_version}")
        print(f"Status Code: {response.status_code}")
        assert response.status_code == 200
        promote_result = response.json()
        assert promote_result["status"] == "success"
        print(f"Successfully verified model promotion for {test_version}!")

    print("\n=========================================")
    print("SUCCESS: ALL ENDPOINTS VERIFIED AND WORKING!")
    print("=========================================")

except Exception as e:
    print(f"Verification Suite Error: {e}")
    sys.exit(1)
