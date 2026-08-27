"""
Manual model retraining script for ChargeGuard.

This script allows on-demand retraining of the fraud detection model
with evaluation and version management.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
import lightgbm as lgb
import shap

# Add parent directory to path for imports
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from app.db import SessionLocal
    from app.models import Transaction, Decision, ModelVersion
except ImportError:
    from db import SessionLocal
    from models import Transaction, Decision, ModelVersion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "app", "chargeback_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "app", "metrics.json")

FEATURE_ORDER = [
    "Amount",
    "tx_count_24h",
    "minutes_since_last_tx",
    "amount_vs_card_avg",
    "is_odd_hour"
]

def load_training_data():
    """
    Load training data from the database.
    In production, this would load historical transactions with outcomes.
    For now, returns synthetic data for demonstration.
    """
    db = SessionLocal()
    
    try:
        # Try to load real data from database
        transactions = db.query(Transaction).all()
        
        if len(transactions) > 100:  # Only use real data if we have enough samples
            print(f"Loading {len(transactions)} transactions from database...")
            
            data = []
            labels = []
            
            for txn in transactions:
                # For demonstration, create synthetic labels based on features
                # In production, these would come from actual chargeback outcomes
                data.append({
                    'Amount': txn.amount,
                    'tx_count_24h': txn.tx_count_24h,
                    'minutes_since_last_tx': txn.minutes_since_last_tx,
                    'amount_vs_card_avg': txn.amount_vs_card_avg,
                    'is_odd_hour': int(txn.is_odd_hour)
                })
                
                # Simple heuristic for synthetic labels (replace with real outcomes)
                fraud_score = (
                    (txn.amount > 100) * 0.3 +
                    (txn.tx_count_24h > 3) * 0.3 +
                    (txn.minutes_since_last_tx < 5) * 0.2 +
                    (txn.amount_vs_card_avg > 0.3) * 0.2
                )
                labels.append(1 if fraud_score > 0.5 else 0)
            
            return pd.DataFrame(data), np.array(labels)
        else:
            print("Not enough data in database, generating synthetic training data...")
            return generate_synthetic_data(n_samples=1000)
            
    finally:
        db.close()

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic training data for demonstration."""
    np.random.seed(42)
    
    data = {
        'Amount': np.random.exponential(scale=50, size=n_samples),
        'tx_count_24h': np.random.poisson(lam=2, size=n_samples),
        'minutes_since_last_tx': np.random.exponential(scale=30, size=n_samples),
        'amount_vs_card_avg': np.random.beta(a=2, b=5, size=n_samples),
        'is_odd_hour': np.random.binomial(n=1, p=0.2, size=n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate synthetic labels based on feature patterns
    fraud_prob = (
        (df['Amount'] > 100) * 0.3 +
        (df['tx_count_24h'] > 3) * 0.3 +
        (df['minutes_since_last_tx'] < 5) * 0.2 +
        (df['amount_vs_card_avg'] > 0.3) * 0.2 +
        (df['is_odd_hour'] == 1) * 0.1
    )
    
    labels = (fraud_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    return df, labels

def train_model(X_train, y_train):
    """Train a LightGBM model."""
    print("Training LightGBM model...")
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    print("Evaluating model...")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall
    }

def calculate_optimal_threshold(y_test, y_pred_proba):
    """Calculate optimal threshold based on cost-benefit analysis."""
    # Simple cost-benefit: assume false positive costs $10, false negative costs $100
    fp_cost = 10
    fn_cost = 100
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    min_cost = float('inf')
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        cost = fp * fp_cost + fn * fn_cost
        
        if cost < min_cost:
            min_cost = cost
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.4f} (cost: ${min_cost:.2f})")
    return best_threshold

def save_model_and_metrics(model, metrics, threshold, version_name):
    """Save model and metrics to files."""
    # Save model
    model_filename = f"chargeback_model_{version_name}.pkl"
    model_path = os.path.join(BASE_DIR, "app", model_filename)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_data = {
        'version': version_name,
        'features': FEATURE_ORDER,
        'threshold': threshold,
        'auc': metrics['auc'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'training_date': datetime.utcnow().isoformat(),
        'model_file': model_filename
    }
    
    metrics_filename = f"metrics_{version_name}.json"
    metrics_path = os.path.join(BASE_DIR, "app", metrics_filename)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    return model_path, metrics_path

def register_model_version(db, version_name, model_path, metrics_path, metrics, threshold):
    """Register the new model version in the database."""
    # Deactivate existing active models
    db.query(ModelVersion).filter(ModelVersion.is_active == True).update({'is_active': False})
    
    # Create new model version
    new_version = ModelVersion(
        version_name=version_name,
        model_file_path=model_path,
        metrics_file_path=metrics_path,
        features=json.dumps(FEATURE_ORDER),
        threshold=threshold,
        auc=metrics['auc'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        training_date=datetime.utcnow(),
        is_active=True
    )
    
    db.add(new_version)
    db.commit()
    
    print(f"Model version {version_name} registered in database")
    return new_version.id

def main():
    """Main retraining pipeline."""
    print("=" * 60)
    print("ChargeGuard Model Retraining Pipeline")
    print("=" * 60)
    
    # Load data
    X, y = load_training_data()
    print(f"Loaded {len(X)} samples with {sum(y)} fraud cases")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Calculate optimal threshold
    threshold = calculate_optimal_threshold(y_test, model.predict_proba(X_test)[:, 1])
    
    # Generate version name
    version_name = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Save model and metrics
    model_path, metrics_path = save_model_and_metrics(model, metrics, threshold, version_name)
    
    # Register in database
    db = SessionLocal()
    try:
        model_version_id = register_model_version(
            db, version_name, model_path, metrics_path, metrics, threshold
        )
        print(f"Model version ID: {model_version_id}")
    finally:
        db.close()
    
    print("=" * 60)
    print("Retraining completed successfully!")
    print(f"New model version: {version_name}")
    print(f"Model path: {model_path}")
    print(f"Metrics path: {metrics_path}")
    print("=" * 60)
    
    # Ask if user wants to promote to production
    promote = input("\nDo you want to promote this model to production? (y/n): ")
    if promote.lower() == 'y':
        # Update main model files
        import shutil
        shutil.copy(model_path, MODEL_PATH)
        shutil.copy(metrics_path, METRICS_PATH)
        print(f"Model promoted to production: {MODEL_PATH}")
        print(f"Metrics promoted to production: {METRICS_PATH}")
    else:
        print("Model saved but not promoted to production.")

if __name__ == "__main__":
    main()
