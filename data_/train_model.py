import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_and_engineer(base_dir, split_name):
    file_path = os.path.join(base_dir, "splits", f"{split_name}.csv")
    df = pd.read_csv(file_path)
    
    # 1. Create features (NO balance errors for realistic simulation)
    # Commented out because these cause unrealistic performance
    # df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    # df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['hour_of_day'] = df['step'] % 24
    df['type_is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    
    # 2. Drop Noise and Leakage
    cols_to_drop = [
        'step', 'type', 'nameOrig', 'nameDest', 'isFlaggedFraud',
        'newbalanceOrig', 'newbalanceDest'
    ]
    df.drop(columns=cols_to_drop, inplace=True)
    
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    
    # Log the features so you can see errorBalance is gone
    logging.info(f"Features being used (Option B): {X.columns.tolist()}")
    
    return X, y


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    logging.info("Loading and engineering splits...")
    X_train, y_train = load_and_engineer(base_dir, "train")
    X_val, y_val = load_and_engineer(base_dir, "validation")
    X_test, y_test = load_and_engineer(base_dir, "test")
    
    logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logging.info(f"Features: {X_train.columns.tolist()}")

    # ==========================================
    # 1. SCALING (Fit on Train ONLY)
    # ==========================================
    logging.info("Fitting StandardScaler on Training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, "scaler.pkl")
    logging.info("Scaler saved as 'scaler.pkl'")

    # ==========================================
    # 2. XGBOOST TRAINING (Handles Imbalance)
    # ==========================================
    neg, pos = y_train.value_counts()
    scale_pos_weight = neg / pos
    logging.info(f"Scale pos weight calculated: {scale_pos_weight:.2f}. Capping at 100 for stability.")

    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        scale_pos_weight=min(scale_pos_weight, 100),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    logging.info("Training XGBoost model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )
    
    joblib.dump(model, "fraud_model.pkl")
    logging.info("Model saved as 'fraud_model.pkl'")

    # ==========================================
    # 3. VALIDATION EVALUATION
    # ==========================================
    logging.info("Evaluating on Validation Set...")
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    print("\n" + "="*60)
    print("VALIDATION SET PERFORMANCE")
    print("="*60)
    print(f"AUC-ROC Score: {roc_auc_score(y_val, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Legit', 'Fraud']))
    
    # ==========================================
    # 4. FINAL TEST EVALUATION (Run this ONCE)
    # ==========================================
    logging.info("Evaluating on HIDDEN Test Set (Final Score)...")
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n" + "="*60)
    print("FINAL TEST SET PERFORMANCE (Report this score)")
    print("="*60)
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_proba_test):.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred_test, target_names=['Legit', 'Fraud']))
    print("="*60)
    print("✅ Pipeline complete! Model and scaler saved for deployment.")

if __name__ == "__main__":
    main()