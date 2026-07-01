import os
from dotenv import load_dotenv

# Load .env from project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)


import json
import joblib
import pandas as pd
import psycopg2
import redis  # <-- ADD
from confluent_kafka import Consumer, Producer, KafkaError


DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "fraud_stream_db"),
    "user": os.getenv("POSTGRES_USER", "fraud_user"),
    "password": os.environ["POSTGRES_PASSWORD"]
}


# ---- PATHS ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data_")

MODEL_PATH = os.path.join(DATA_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(DATA_DIR, "threshold.json")

# Load model assets
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = json.load(f)["threshold"]

# Detect number of features
n_features = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else 5
print(f"Model loaded. Threshold: {THRESHOLD}")
print(f"   Model expects {n_features} features.")

# ---- Redis Connection (for user context) ----
try:
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    r.ping()
    print("Connected to Redis for user context.")
except Exception as e:
    print(f"Redis unavailable: {e}")
    r = None

# ---- Kafka Consumer ----
consumer_conf = {
    'bootstrap.servers': '127.0.0.1:9092',
    'group.id': 'fraud-detection-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': True,
    'session.timeout.ms': 6000,
}
consumer = Consumer(consumer_conf)
consumer.subscribe(['transactions'])

# ---- Kafka Producer ----
producer_conf = {'bootstrap.servers': '127.0.0.1:9092'}
producer = Producer(producer_conf)

#
print("Listening for transactions... (press Ctrl+C to stop)")
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print(f"Consumer error: {msg.error()}")
            continue

        tx = json.loads(msg.value().decode('utf-8'))

        # ---- 1. Fetch Redis context (velocity, avg) ----
        velocity_1m = 0
        avg_amount_1h = 0.0
        user = tx.get('nameOrig', tx.get('account_number', 'UNKNOWN'))
        if r:
            ctx_raw = r.get(f"user:{user}")
            if ctx_raw:
                try:
                    ctx = json.loads(ctx_raw)
                    velocity_1m = ctx.get('velocity_1m', 0)
                    avg_amount_1h = ctx.get('avg_amount_1h', 0.0)
                except:
                    pass

        # ---- 2. Feature Engineering ----
        hour_of_day = tx.get('step', 0) % 24
        type_is_transfer = 1 if tx.get('transaction_type', '').upper() == "TRANSFER" else 0

        # ---- 3. Build feature vector ----
        if n_features == 5:
            input_data = pd.DataFrame([[
                tx['amount'],
                tx['oldbalanceOrg'],
                tx['oldbalanceDest'],
                hour_of_day,
                type_is_transfer
            ]], columns=['amount', 'oldbalanceOrg', 'oldbalanceDest', 'hour_of_day', 'type_is_transfer'])
            context_used = False
        elif n_features == 7:
            input_data = pd.DataFrame([[
                tx['amount'],
                tx['oldbalanceOrg'],
                tx['oldbalanceDest'],
                hour_of_day,
                type_is_transfer,
                velocity_1m,
                avg_amount_1h
            ]], columns=['amount', 'oldbalanceOrg', 'oldbalanceDest', 'hour_of_day', 'type_is_transfer',
                         'velocity_1m', 'avg_amount_1h'])
            context_used = True
        else:
            # fallback to 5 features
            input_data = pd.DataFrame([[
                tx['amount'],
                tx['oldbalanceOrg'],
                tx['oldbalanceDest'],
                hour_of_day,
                type_is_transfer
            ]], columns=['amount', 'oldbalanceOrg', 'oldbalanceDest', 'hour_of_day', 'type_is_transfer'])
            context_used = False

        X_scaled = scaler.transform(input_data)
        prob = float(model.predict_proba(X_scaled)[0, 1])
        action = "BLOCKED" if prob >= THRESHOLD else "APPROVED"

        # ---- 4. Prepare result ----
        result = {
            "transaction_id": tx['transaction_id'],
            "fraud_probability": round(prob, 4),
            "account_number": user,
            "amount": tx.get('amount', 0.0),
            "action": action,
            "threshold_used": THRESHOLD
        }

        # ---- 5. Insert into PostgreSQL (including raw features for SHAP) ----
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO predictions (
                    transaction_id, fraud_probability, account_number,
                    amount, action, threshold_used,
                    oldbalanceOrg, oldbalanceDest, step, transaction_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                result['transaction_id'],
                result['fraud_probability'],
                result['account_number'],
                result['amount'],
                result['action'],
                result['threshold_used'],
                tx.get('oldbalanceOrg', 0.0),
                tx.get('oldbalanceDest', 0.0),
                tx.get('step', 0),
                tx.get('transaction_type', 'UNKNOWN')
            ))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"⚠️ DB insert failed: {e}")

        # ---- 6. Send to fraud_results ----
        producer.produce('fraud_results', value=json.dumps(result).encode('utf-8'))
        producer.flush()
        print(f"{result['transaction_id']} | Prob: {prob:.2f} | Action: {action} | User: {user} | Context used: {context_used}")

except KeyboardInterrupt:
    print("Shutting down...")
finally:
    consumer.close()