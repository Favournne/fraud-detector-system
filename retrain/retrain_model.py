import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
import os

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_")

SPLIT_DIR = os.path.join(DATA_DIR, "splits")
MODEL_PATH = os.path.join(DATA_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(DATA_DIR, "threshold.json")

# ---------- LOAD SPLITS ----------
print("Loading splits...")
train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(SPLIT_DIR, "validation.csv"))
test_df  = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))

print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

# ---------- FUNCTION: COMPUTE ROLLING FEATURES FOR A DATAFRAME ----------
def compute_rolling_features(df):
    # Sort by step to preserve chronological order
    df = df.sort_values('step').reset_index(drop=True)
    user_buffers = {}
    velocity_1m = []
    avg_amount_1h = []

    for idx, row in df.iterrows():
        user = row['nameOrig']
        step = row['step']
        amount = row['amount']

        buf = user_buffers.setdefault(user, {'steps': [], 'amounts': []})

        # Prune older than 1 step (last hour)
        cutoff = step - 1
        while buf['steps'] and buf['steps'][0] < cutoff:
            buf['steps'].pop(0)
            buf['amounts'].pop(0)

        # Count transactions in the last step (last hour)
        vel = 0
        cutoff_1m = step - 1
        for s in buf['steps']:
            if s >= cutoff_1m:
                vel += 1

        # Average amount over the last step
        sum_1h = sum(buf['amounts'])
        count_1h = len(buf['amounts'])
        avg_1h = sum_1h / count_1h if count_1h > 0 else 0.0

        velocity_1m.append(vel)
        avg_amount_1h.append(round(avg_1h, 2))

        # Append current transaction
        buf['steps'].append(step)
        buf['amounts'].append(amount)

    df['velocity_1m'] = velocity_1m
    df['avg_amount_1h'] = avg_amount_1h
    return df

# ---------- APPLY ROLLING FEATURES TO EACH SPLIT ----------
print("Computing rolling features...")
train_df = compute_rolling_features(train_df)
val_df   = compute_rolling_features(val_df)
test_df  = compute_rolling_features(test_df)

# ---------- FEATURE ENGINEERING (same for all splits) ----------
def engineer_features(df):
    df['hour_of_day'] = df['step'] % 24
    df['type_is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    # Drop leakage and high-cardinality columns
    cols_to_drop = ['step', 'type', 'nameOrig', 'nameDest', 'isFlaggedFraud',
                    'newbalanceOrig', 'newbalanceDest']
    df.drop(columns=cols_to_drop, inplace=True)
    return df

print("Engineering features...")
train_df = engineer_features(train_df)
val_df   = engineer_features(val_df)
test_df  = engineer_features(test_df)

# ---------- SPLIT FEATURES AND TARGET ----------
feature_cols = ['amount', 'oldbalanceOrg', 'oldbalanceDest',
                'hour_of_day', 'type_is_transfer',
                'velocity_1m', 'avg_amount_1h']

X_train = train_df[feature_cols]
y_train = train_df['isFraud']
X_val   = val_df[feature_cols]
y_val   = val_df['isFraud']
X_test  = test_df[feature_cols]
y_test  = test_df['isFraud']

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

# ---------- SCALE (fit on train only) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ---------- TRAIN XGBOOST ----------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
print("Training XGBoost...")
model.fit(X_train_scaled, y_train)

# ---------- THRESHOLD SELECTION ON VALIDATION SET (precision-floor rule) ----------
y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)

# Find thresholds where precision >= 0.50
candidates = [(t, p, r) for t, p, r in zip(thresholds, precisions[:-1], recalls[:-1]) if p >= 0.50]
if candidates:
    # Choose the one with highest recall (lowest threshold among those meeting precision)
    best_threshold = max(candidates, key=lambda x: x[2])[0]
else:
    # Fallback: max F1
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
print(f"Selected threshold: {best_threshold:.4f}")

# ---------- EVALUATE ON TEST SET ----------
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_test_proba >= best_threshold).astype(int)

test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Test Set Performance ---")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print("Confusion Matrix:")
print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

# ---------- SAVE ASSETS ----------
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
with open(THRESHOLD_PATH, 'w') as f:
    json.dump({"threshold": float(best_threshold)}, f)

print(f"\n Model saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")
print(f"✅ Threshold saved to {THRESHOLD_PATH}")