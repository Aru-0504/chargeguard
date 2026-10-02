"""
Model retraining and version management pipeline for ChargeGuard.

Features:
- Supports training on real historical chargeback dataset (ml/data/processed/chargeback_features.csv or ml/data/raw/df.csv),
  database transactions, or synthetic fallback.
- Optimizes decision threshold via financial cost-curve minimization (FP cost vs FN cost).
- Evaluates AUC, Precision, Recall, and financial impact.
- Generates versioned artifacts (chargeback_model_{version}.pkl, metrics_{version}.json).
- Registers versions in database model_versions table.
- Supports automated promotion via CLI flags (--promote, --no-promote) or interactive prompt.
"""
import os
import sys
import json
import argparse
import shutil
import warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    classification_report
)
import lightgbm as lgb
from dotenv import load_dotenv

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, ".env"))
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "sqlite:///./test.db"

try:
    from app.db import SessionLocal, init_db
    from app.models import Transaction, Decision, ModelVersion
except ImportError:
    from db import SessionLocal, init_db
    from models import Transaction, Decision, ModelVersion

# Initialize database tables
init_db()

PROD_MODEL_PATH = os.path.join(BASE_DIR, "app", "chargeback_model.pkl")
PROD_METRICS_PATH = os.path.join(BASE_DIR, "app", "metrics.json")
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "ml", "data", "processed", "chargeback_features.csv")
RAW_DATA_PATH = os.path.join(PROJECT_DIR, "ml", "data", "raw", "df.csv")

FEATURE_ORDER = [
    "Amount",
    "tx_count_24h",
    "minutes_since_last_tx",
    "amount_vs_card_avg",
    "is_odd_hour"
]


def engineer_features_from_raw(raw_path: str) -> pd.DataFrame:
    """Engineer features from raw transaction log."""
    print(f"Engineering features from raw data: {raw_path}")
    df = pd.read_csv(raw_path, parse_dates=['Date'])
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df['target'] = (df['CBK'] == 'Yes').astype(int)
    df = df.sort_values(['Card Number', 'Date']).reset_index(drop=True)

    # 1. Transaction count in past 24 hours per card
    df['tx_count_24h'] = df.groupby('Card Number')['Date'].transform(
        lambda s: s.apply(lambda t: ((s >= t - pd.Timedelta(hours=24)) & (s < t)).sum())
    )

    # 2. Minutes since card's previous transaction (-1 for card's first transaction)
    df['prev_tx_time'] = df.groupby('Card Number')['Date'].shift(1)
    df['minutes_since_last_tx'] = (df['Date'] - df['prev_tx_time']).dt.total_seconds() / 60
    df['minutes_since_last_tx'] = df['minutes_since_last_tx'].fillna(-1)

    # 3. Spend deviation relative to historical card average
    df['card_avg_amount'] = df.groupby('Card Number')['Amount'].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    df['amount_vs_card_avg'] = (df['Amount'] / df['card_avg_amount'].replace(0, np.nan)) - 1
    df['amount_vs_card_avg'] = df['amount_vs_card_avg'].fillna(0)

    # 4. Odd hour flag (before 6am or after 10pm)
    df['hour'] = df['Date'].dt.hour
    df['is_odd_hour'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)

    return df


def generate_synthetic_data(n_samples: int = 1500):
    """Generate synthetic fallback dataset."""
    np.random.seed(42)
    data = {
        'Amount': np.random.exponential(scale=50, size=n_samples),
        'tx_count_24h': np.random.poisson(lam=2, size=n_samples),
        'minutes_since_last_tx': np.random.exponential(scale=30, size=n_samples),
        'amount_vs_card_avg': np.random.beta(a=2, b=5, size=n_samples),
        'is_odd_hour': np.random.binomial(n=1, p=0.2, size=n_samples)
    }
    df = pd.DataFrame(data)
    fraud_prob = (
        (df['Amount'] > 100) * 0.3 +
        (df['tx_count_24h'] > 3) * 0.3 +
        (df['minutes_since_last_tx'] < 5) * 0.2 +
        (df['amount_vs_card_avg'] > 0.3) * 0.2 +
        (df['is_odd_hour'] == 1) * 0.1
    )
    labels = (fraud_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    return df[FEATURE_ORDER], labels


def load_training_data(source: str = "auto", n_synthetic: int = 1500):
    """
    Load training data according to requested source.
    Sources: 'auto', 'processed', 'raw', 'db', 'synthetic'
    """
    source = source.lower()

    # 1. Processed CSV
    if source in ("auto", "processed") and os.path.exists(PROCESSED_DATA_PATH):
        print(f"Loading pre-engineered dataset from: {PROCESSED_DATA_PATH}")
        df = pd.read_csv(PROCESSED_DATA_PATH)
        if all(col in df.columns for col in FEATURE_ORDER + ['target']):
            X = df[FEATURE_ORDER]
            y = df['target'].to_numpy()
            return X, y, f"processed_csv ({len(X)} records)"

    # 2. Raw CSV
    if source in ("auto", "raw") and os.path.exists(RAW_DATA_PATH):
        df = engineer_features_from_raw(RAW_DATA_PATH)
        X = df[FEATURE_ORDER]
        y = df['target'].to_numpy()
        return X, y, f"raw_csv_engineered ({len(X)} records)"

    # 3. Database
    if source in ("auto", "db"):
        db = SessionLocal()
        try:
            transactions = db.query(Transaction).all()
            if len(transactions) >= 100:
                print(f"Loading {len(transactions)} historical transactions from database...")
                data = []
                labels = []
                ground_truth_count = 0
                for txn in transactions:
                    data.append({
                        'Amount': txn.amount,
                        'tx_count_24h': txn.tx_count_24h,
                        'minutes_since_last_tx': txn.minutes_since_last_tx,
                        'amount_vs_card_avg': txn.amount_vs_card_avg,
                        'is_odd_hour': int(txn.is_odd_hour)
                    })
                    # Check for verified dispute outcome on corresponding decision
                    dec = db.query(Decision).filter(Decision.transaction_id == txn.id).first()
                    outcome = getattr(dec, "dispute_outcome", None) if dec else None
                    if outcome in ("lost", "won"):
                        # 'lost' dispute confirms true fraud/chargeback (label=1)
                        # 'won' dispute confirms false chargeback claim (label=0)
                        labels.append(1 if outcome == "lost" else 0)
                        ground_truth_count += 1
                    else:
                        score = (
                            (txn.amount > 100) * 0.3 +
                            (txn.tx_count_24h > 3) * 0.3 +
                            (txn.minutes_since_last_tx < 5) * 0.2 +
                            (txn.amount_vs_card_avg > 0.3) * 0.2
                        )
                        labels.append(1 if score > 0.5 else 0)

                if ground_truth_count > 0:
                    print(f"Loaded {ground_truth_count} real dispute outcome labels from feedback loop.")
                return pd.DataFrame(data), np.array(labels), f"db_transactions ({len(transactions)} records, {ground_truth_count} verified outcomes)"
        except Exception as e:
            print(f"DB load failed: {e}")
        finally:
            db.close()

    # 4. Fallback Synthetic
    print(f"Using synthetic dataset fallback ({n_synthetic} samples)...")
    X, y = generate_synthetic_data(n_samples=n_synthetic)
    return X, y, f"synthetic ({n_synthetic} samples)"


def train_model(X_train: pd.DataFrame, y_train: np.ndarray) -> lgb.LGBMClassifier:
    """Train LightGBM classifier with class imbalance weighting."""
    print("Training LightGBM classifier...")
    model = lgb.LGBMClassifier(
        class_weight='balanced',
        learning_rate=0.05,
        max_depth=5,
        n_estimators=200,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    return model


def calculate_optimal_threshold(
    y_test: np.ndarray,
    probs: np.ndarray,
    fp_cost: float = 50.0,
    fn_cost: float = 183.3
):
    """
    Search precision-recall threshold curve to find the threshold minimizing financial loss:
    Total Cost = FP * FP_COST + FN * FN_COST
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    best_cost = float('inf')
    best_t = 0.5
    total_positives = y_test.sum()

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        tp = r * total_positives
        fn = total_positives - tp
        fp = (tp / p - tp) if p > 0 else 0
        cost = fp * fp_cost + fn * fn_cost
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)

    return best_t, best_cost


def evaluate_model(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    fp_cost: float = 50.0,
    fn_cost: float = 183.3
):
    """Evaluate model with AUC, optimal threshold, precision, and recall."""
    probs = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probs))

    # Calculate optimal threshold
    best_threshold, best_cost = calculate_optimal_threshold(y_test, probs, fp_cost=fp_cost, fn_cost=fn_cost)
    preds = (probs >= best_threshold).astype(int)

    precision = float(precision_score(y_test, preds, zero_division=0))
    recall = float(recall_score(y_test, preds, zero_division=0))

    print(f"\n--- Model Evaluation ---")
    print(f"AUC:                 {auc:.4f}")
    print(f"Optimal Threshold:   {best_threshold:.4f}")
    print(f"Precision @Thresh:   {precision:.4f}")
    print(f"Recall @Thresh:      {recall:.4f}")
    print(f"Estimated Cost:      INR {best_cost:.2f} (FP Cost: INR {fp_cost}, FN Cost: INR {fn_cost})")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    return {
        "auc": round(auc, 4),
        "threshold": round(best_threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "estimated_total_cost": round(best_cost, 2),
        "fp_cost_assumption": fp_cost,
        "fn_cost_assumption": fn_cost,
        "test_set_size": len(X_test)
    }


def save_model_artifacts(
    model: lgb.LGBMClassifier,
    eval_metrics: dict,
    version_name: str,
    data_source: str
):
    """Save versioned model and metrics artifacts."""
    model_filename = f"chargeback_model_{version_name}.pkl"
    metrics_filename = f"metrics_{version_name}.json"

    model_path = os.path.join(BASE_DIR, "app", model_filename)
    metrics_path = os.path.join(BASE_DIR, "app", metrics_filename)

    # Save model
    joblib.dump(model, model_path)
    print(f"Versioned model saved to: {model_path}")

    # Save metrics metadata
    metrics_payload = {
        "version": version_name,
        "features": FEATURE_ORDER,
        "threshold": eval_metrics["threshold"],
        "auc": eval_metrics["auc"],
        "precision": eval_metrics["precision"],
        "recall": eval_metrics["recall"],
        "fp_cost_assumption": eval_metrics["fp_cost_assumption"],
        "fn_cost_assumption": eval_metrics["fn_cost_assumption"],
        "estimated_total_cost": eval_metrics["estimated_total_cost"],
        "test_set_size": eval_metrics["test_set_size"],
        "data_source": data_source,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "model_file": model_filename
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Versioned metrics saved to: {metrics_path}")

    return model_path, metrics_path, metrics_payload


def register_model_version(
    version_name: str,
    model_path: str,
    metrics_path: str,
    metrics: dict,
    is_active: bool = False
):
    """Register version in DB."""
    db = SessionLocal()
    try:
        if is_active:
            db.query(ModelVersion).filter(ModelVersion.is_active == True).update({'is_active': False})

        new_version = ModelVersion(
            version_name=version_name,
            model_file_path=model_path,
            metrics_file_path=metrics_path,
            features=json.dumps(FEATURE_ORDER),
            threshold=metrics['threshold'],
            auc=metrics['auc'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            training_date=datetime.now(timezone.utc),
            is_active=is_active
        )
        db.add(new_version)
        db.commit()
        db.refresh(new_version)
        print(f"Registered model version '{version_name}' in database (ID: {new_version.id}, is_active={is_active})")
        return new_version.id
    except Exception as e:
        print(f"Warning: Could not register model in database: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def promote_to_production(model_path: str, metrics_path: str, version_name: str):
    """Promote artifacts to production paths and mark active in DB."""
    shutil.copy(model_path, PROD_MODEL_PATH)
    shutil.copy(metrics_path, PROD_METRICS_PATH)
    print(f"\n[PROMOTION] Successfully promoted '{version_name}' to production:")
    print(f"  -> {PROD_MODEL_PATH}")
    print(f"  -> {PROD_METRICS_PATH}")

    db = SessionLocal()
    try:
        db.query(ModelVersion).update({ModelVersion.is_active: False})
        record = db.query(ModelVersion).filter(ModelVersion.version_name == version_name).first()
        if record:
            record.is_active = True
        db.commit()
        print(f"[PROMOTION] Database is_active flag set for version '{version_name}'")
    except Exception as e:
        print(f"Warning: Could not update active flag in database: {e}")
        db.rollback()
    finally:
        db.close()


def run_pipeline(
    data_source: str = "auto",
    promote: bool = None,
    fp_cost: float = 50.0,
    fn_cost: float = 183.3,
    n_synthetic: int = 1500
):
    print("=" * 65)
    print("  ChargeGuard ML Retraining & Version Management Pipeline")
    print("=" * 65)

    # 1. Load data
    X, y, source_desc = load_training_data(source=data_source, n_synthetic=n_synthetic)
    print(f"Dataset: {source_desc} | Fraud rate: {y.mean() * 100:.2f}% ({y.sum()} / {len(y)})")

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")

    # 3. Train Model
    model = train_model(X_train, y_train)

    # 4. Evaluate & Calculate Cost-Optimal Threshold
    metrics = evaluate_model(model, X_test, y_test, fp_cost=fp_cost, fn_cost=fn_cost)

    # 5. Save Artifacts
    version_name = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    model_path, metrics_path, metrics_payload = save_model_artifacts(
        model, metrics, version_name, source_desc
    )

    # 6. Register in DB
    register_model_version(
        version_name=version_name,
        model_path=model_path,
        metrics_path=metrics_path,
        metrics=metrics,
        is_active=False
    )

    # 7. Promotion decision
    should_promote = False
    if promote is True:
        should_promote = True
    elif promote is False:
        should_promote = False
    elif sys.stdin.isatty():
        try:
            choice = input(f"\nDo you want to promote {version_name} to production? (y/N): ").strip().lower()
            should_promote = (choice == "y")
        except EOFError:
            should_promote = False
    else:
        print("\nNon-interactive shell: model saved as versioned candidate (not promoted).")

    if should_promote:
        promote_to_production(model_path, metrics_path, version_name)
    else:
        print(f"\nCandidate version '{version_name}' is ready. Promote anytime via API or CLI.")

    print("=" * 65)
    return {
        "version": version_name,
        "metrics": metrics,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "promoted": should_promote
    }


def main():
    parser = argparse.ArgumentParser(description="ChargeGuard Model Retraining Pipeline")
    parser.add_argument(
        "--data-source",
        choices=["auto", "processed", "raw", "db", "synthetic"],
        default="auto",
        help="Source of training data (default: auto)"
    )
    parser.add_argument(
        "--promote",
        dest="promote",
        action="store_true",
        default=None,
        help="Automatically promote retrained model to production"
    )
    parser.add_argument(
        "--no-promote",
        dest="promote",
        action="store_false",
        help="Do not promote to production"
    )
    parser.add_argument(
        "--fp-cost",
        type=float,
        default=50.0,
        help="False positive friction cost assumption (default: 50.0)"
    )
    parser.add_argument(
        "--fn-cost",
        type=float,
        default=183.3,
        help="False negative chargeback loss assumption (default: 183.3)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1500,
        help="Sample count if using synthetic data (default: 1500)"
    )

    args = parser.parse_args()
    run_pipeline(
        data_source=args.data_source,
        promote=args.promote,
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
        n_synthetic=args.samples
    )


if __name__ == "__main__":
    main()
