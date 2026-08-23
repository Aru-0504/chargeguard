import sys
import os

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
        print("Success: Successfully connected to Neon PostgreSQL!")
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
      "is_odd_hour": True
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
    
    print("\n=========================================")
    print("SUCCESS: ALL ENDPOINTS VERIFIED AND WORKING!")
    print("=========================================")

except Exception as e:
    print(f"Verification Suite Error: {e}")
    sys.exit(1)
