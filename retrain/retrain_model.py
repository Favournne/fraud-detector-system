# retrain_model.py
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
import os

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_")

CSV_PATH = os.path.join(DATA_DIR, "paysim.csv")
MODEL_PATH = os.path.join(DATA_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(DATA_DIR, "threshold.json")

print(f"Looking for CSV at: {CSV_PATH}")

# ---------- LOAD DATA ----------
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} rows.")
df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
print(f"Filtered to {len(df):,} transactions.")
df = df.sort_values('step').reset_index(drop=True)

# ---------- COMPUTE ROLLING FEATURES ----------
user_buffers = {}
velocity_1m = []
avg_amount_1h = []

for idx, row in df.iterrows():
    user = row['nameOrig']
    step = row['step']
    amount = row['amount']
    buf = user_buffers.setdefault(user, {'steps': [], 'amounts': []})
    
    cutoff = step - 3600
    while buf['steps'] and buf['steps'][0] < cutoff:
        buf['steps'].pop(0)
        buf['amounts'].pop(0)
    
    vel = 0
    cutoff_1m = step - 60
    for s, a in zip(buf['steps'], buf['amounts']):
        if s >= cutoff_1m:
            vel += 1
    
    sum_1h = sum(buf['amounts'])
    count_1h = len(buf['amounts'])
    avg_1h = sum_1h / count_1h if count_1h > 0 else 0.0
    
    velocity_1m.append(vel)
    avg_amount_1h.append(round(avg_1h, 2))
    
    buf['steps'].append(step)
    buf['amounts'].append(amount)

df['velocity_1m'] = velocity_1m
df['avg_amount_1h'] = avg_amount_1h
print("Rolling features computed.")

# ---------- FEATURE ENGINEERING ----------
df['hour_of_day'] = df['step'] % 24
df['type_is_transfer'] = (df['type'] == 'TRANSFER').astype(int)

feature_cols = [
    'amount', 'oldbalanceOrg', 'oldbalanceDest',
    'hour_of_day', 'type_is_transfer',
    'velocity_1m', 'avg_amount_1h'
]
X = df[feature_cols].values
y = df['isFraud'].values
print(f"Feature shape: {X.shape}")

# ---------- TRAIN/TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- SCALE ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
model.fit(X_train_scaled, y_train)

# ---------- FIND OPTIMAL THRESHOLD ----------
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
print(f"Optimal threshold (F1): {best_threshold:.4f}")

# ---------- EVALUATE ----------
y_pred = (y_pred_proba >= best_threshold).astype(int)
test_f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Test F1 score:  {test_f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print("Confusion Matrix:")
print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

# ---------- SAVE ASSETS ----------
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
with open(THRESHOLD_PATH, 'w') as f:
    json.dump({"threshold": float(best_threshold)}, f)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")
print(f"✅ Threshold saved to {THRESHOLD_PATH}")