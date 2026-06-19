"""
retrain.py
----------
Pulls labeled data from fraud_events.db, retrains the fraud detection
pipeline, compares against the current production model, and replaces
it only if the new model wins on F1 score.

Every run is tracked with MLflow.

Run with: python retrain.py
"""

import os
import shutil
import sqlite3
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import FraudFeatureEngineer

#CONFIGURATION

DB_PATH = r"C:\Users\USER\fraud_project\model_\fraud_events.db"
MODEL_DIR = r"C:\Users\USER\fraud_project\model_\models"
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, "fraud_pipeline_v1.0.pkl")
MLFLOW_EXPERIMENT = "fraud_detection_retraining"
OPERATIONAL_THRESHOLD = 0.75

# Minimum records before retraining is worth attempting
MIN_RECORDS_REQUIRED = 500

# Challenger must beat current model by this margin to be deployed
MIN_IMPROVEMENT_THRESHOLD = 0.01  # 1% F1 improvement



# STEP 1 — PULL DATA FROM SQLITE

# Place this at the very start of your main() function
os.makedirs(MODEL_DIR, exist_ok=True)

def load_training_data() -> pd.DataFrame:
    print("Pulling labeled data from fraud_events.db...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
            timestamp,
            account_id          AS account_number,
            amount,
            channel,
            transaction_type,
            location            AS location_state,
            device_id,
            destination_bank,
            user_tx_count_1h,
            device_tx_count_1h,
            unique_dest_banks_1h,
            accounts_per_device_24h,
            amount_vs_avg_7d,
            is_anomaly          AS is_fraud
        FROM events
        WHERE action_required IN ('APPROVED', 'BLOCKED')
        """,
        conn,
    )
    conn.close()
    print(f"Loaded {len(df):,} records from database.")
    return df



# STEP 2 — EVALUATE ANY PIPELINE ON TEST DATA

def evaluate_pipeline(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Runs pipeline.predict_proba() and returns AUC + F1.
    Works for both the current production pipeline and the challenger
    since both are sklearn Pipeline objects with the same interface.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= OPERATIONAL_THRESHOLD).astype(int)
    return {
        "auc": round(roc_auc_score(y_test, y_proba), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }



# STEP 3 — BUILD AND TRAIN A CHALLENGER PIPELINE
# Mirrors trainer.build_and_fit() from your original training script exactly

def build_challenger_pipeline(
    X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float
) -> Pipeline:
    """
    Constructs a fresh FraudFeatureEngineer + XGBClassifier pipeline
    in the same structure as the original fraud_pipeline_v1.0.pkl.
    """
    engineer = FraudFeatureEngineer(m_smoothing=100.0)

    # Run secure out-of-fold fit_transform on training data
    X_train_transformed = engineer.fit_transform(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.5,
        reg_lambda=5.0,
        random_state=42,
        tree_method="hist",
        eval_metric="logloss",
    )
    xgb.fit(X_train_transformed, y_train)

    # Bundle into identical Pipeline structure as production model
    challenger = Pipeline([
        ("engineer", engineer),
        ("classifier", xgb),
    ])

    return challenger


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("\n" + "=" * 60)
    print(f"  Retraining Run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # --- Load data ---
    df = load_training_data()

    if len(df) < MIN_RECORDS_REQUIRED:
        print(
            f"\nInsufficient data: {len(df):,} records found, "
            f"minimum is {MIN_RECORDS_REQUIRED:,}.\n"
            "Keep the pipeline running to accumulate more labeled transactions."
        )
        return

    # --- Class distribution check ---
    fraud_count = int(df["is_fraud"].sum())
    legit_count = len(df) - fraud_count
    print(f"\nClass distribution — Fraud: {fraud_count:,} | Legitimate: {legit_count:,}")

    if fraud_count < 10:
        print("Too few fraud examples to retrain meaningfully.")
        return

    # --- Compute class weight dynamically from live data ---
    scale_pos_weight = round(legit_count / fraud_count, 2)
    print(f"Computed scale_pos_weight: {scale_pos_weight}")

    # --- Drop columns the original trainer dropped ---
    drop_cols = ["is_fraud", "transaction_id", "customer_name", "narration"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["is_fraud"]

    # --- Chronological split — no shuffle, preserves time ordering ---
    split_idx = int(len(df) * 0.80)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print(f"Train: {len(X_train):,} records | Test: {len(X_test):,} records")

    # --- Load current production model ---
    print(f"\nLoading current production model...")
    try:
        current_pipeline = joblib.load(CURRENT_MODEL_PATH)
    except Exception as e:
        print(f"Could not load current model: {e}\nAborting.")
        return

    # --- Evaluate current model ---
    print("Evaluating current production model on test set...")
    try:
        current_scores = evaluate_pipeline(current_pipeline, X_test, y_test)
        print(
            f"Current model  —  AUC: {current_scores['auc']}  "
            f"|  F1: {current_scores['f1']}"
        )
    except Exception as e:
        print(f"Could not evaluate current model: {e}")
        current_scores = {"auc": 0.0, "f1": 0.0}

    # --- Train and evaluate challenger ---
    print("\nTraining challenger pipeline on live data...")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(
        run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        # Log dataset context
        mlflow.log_param("train_records", len(X_train))
        mlflow.log_param("test_records", len(X_test))
        mlflow.log_param("fraud_count", fraud_count)
        mlflow.log_param("legit_count", legit_count)
        mlflow.log_param("fraud_rate", round(fraud_count / len(df), 4))
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("operational_threshold", OPERATIONAL_THRESHOLD)

        # Build challenger
        challenger_pipeline = build_challenger_pipeline(X_train, y_train, scale_pos_weight)

        # Evaluate challenger
        print("Evaluating challenger pipeline...")
        challenger_scores = evaluate_pipeline(challenger_pipeline, X_test, y_test)
        print(
            f"Challenger model —  AUC: {challenger_scores['auc']}  "
            f"|  F1: {challenger_scores['f1']}"
        )

        # Full classification report
        y_proba = challenger_pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= OPERATIONAL_THRESHOLD).astype(int)
        print("\nClassification Report (Challenger):")
        print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

        # Log all metrics
        mlflow.log_metric("current_auc", current_scores["auc"])
        mlflow.log_metric("current_f1", current_scores["f1"])
        mlflow.log_metric("challenger_auc", challenger_scores["auc"])
        mlflow.log_metric("challenger_f1", challenger_scores["f1"])
        improvement = round(challenger_scores["f1"] - current_scores["f1"], 4)
        mlflow.log_metric("f1_improvement", improvement)

        # --- Deploy or retain ---
        if improvement >= MIN_IMPROVEMENT_THRESHOLD:
            print(
                f"\nChallenger wins by {improvement:.4f} F1 — deploying to production."
            )

            # Back up current model with timestamp
            backup_path = CURRENT_MODEL_PATH.replace(
                ".pkl",
                f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            )
            shutil.copy(CURRENT_MODEL_PATH, backup_path)
            print(f"Current model backed up to: {backup_path}")

            # Overwrite production model
            joblib.dump(challenger_pipeline, CURRENT_MODEL_PATH)
            print(f"New model deployed to: {CURRENT_MODEL_PATH}")

            mlflow.log_param("deployment_decision", "DEPLOYED")
            mlflow.log_param("backup_path", backup_path)
            mlflow.sklearn.log_model(challenger_pipeline, "challenger_pipeline")

        else:
            print(
                f"\nChallenger did not improve sufficiently "
                f"(delta: {improvement:.4f} F1). Current model retained."
            )
            mlflow.log_param("deployment_decision", "RETAINED")

        print(f"\nMLflow run logged. Run 'mlflow ui' to view experiment history.")

    print("\n" + "=" * 60)
    print("  Retraining run complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()