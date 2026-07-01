import os
import json
import joblib
import pandas as pd
import psycopg2
from kafka import KafkaConsumer, KafkaProducer

# ---- PATHS TO MODEL FILES (in data_ folder, one level up) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # backend/
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data_")    # ../data_

MODEL_PATH = os.path.join(DATA_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(DATA_DIR, "threshold.json")

# 1. Load your model assets
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = json.load(f)["threshold"]

print(f"Model loaded from {DATA_DIR}. Threshold: {THRESHOLD}")

# 2. Kafka Consumer (listens for transactions)
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='127.0.0.1:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='fraud-detection-group',
    api_version_auto_timeout_ms=30000,    # increase
    request_timeout_ms=30000,
    session_timeout_ms=15000,
    max_poll_interval_ms=300000
)

# 3. Kafka Producer (to send results back)
result_producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# 4. PostgreSQL connection details (replace 'your_password' with the actual one)
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "fraud_stream_db",
    "user": "fraud_user",
    "password": "pasiword227"   # <-- UPDATE THIS!
}

print("Fraud detection consumer is running. Listening for transactions...")

for message in consumer:
    tx = message.value

    # Engineer features (exactly like train_model.py)
    hour_of_day = tx.get('step', 0) % 24
    type_is_transfer = 1 if tx.get('transaction_type', '').upper() == "TRANSFER" else 0

    input_data = pd.DataFrame([[
        tx['amount'],
        tx['oldbalanceOrg'],
        tx['oldbalanceDest'],
        hour_of_day,
        type_is_transfer
    ]], columns=['amount', 'oldbalanceOrg', 'oldbalanceDest', 'hour_of_day', 'type_is_transfer'])

    X_scaled = scaler.transform(input_data)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    action = "BLOCKED" if prob >= THRESHOLD else "APPROVED"

    result = {
        "transaction_id": tx['transaction_id'],
        "fraud_probability": round(prob, 4),
        "account_number": tx.get('account_number', 'UNKNOWN'),
        "amount": tx.get('amount', 0.0),
        "action": action,
        "threshold_used": THRESHOLD
    }

    # ---- INSERT INTO POSTGRESQL ----
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                transaction_id, fraud_probability, account_number,
                amount, action, threshold_used
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            result['transaction_id'],
            result['fraud_probability'],
            result['account_number'],
            result['amount'],
            result['action'],
            result['threshold_used']
        ))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"{result['transaction_id']} saved to DB.")
    except Exception as e:
        print(f"DB insert failed for {result['transaction_id']}: {e}")

    # ---- SEND RESULT TO KAFKA ----
    result_producer.send('fraud_results', value=result)
    print(f"{tx['transaction_id']} | Prob: {prob:.2f} | Action: {action}")