import os
from dotenv import load_dotenv

# Load .env from the project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)

import joblib
import pandas as pd
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import psycopg2
import psycopg2.extras
from confluent_kafka import Producer
import shap

# 1. Configuration from environment
print("Loading main.py from:", os.path.abspath(__file__))

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "fraud_stream_db"),
    "user": os.getenv("POSTGRES_USER", "fraud_user"),
    "password": os.environ["POSTGRES_PASSWORD"]
}

REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": 0,
    "decode_responses": True
}

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")


# 2. Kafka Producer

producer_conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'client.id': 'fastapi-producer'
}
kafka_producer = Producer(producer_conf)

def delivery_report(err, msg):
    if err:
        print(f"Kafka delivery failed: {err}")
    else:
        print(f"Kafka delivered to {msg.topic()}")


# 3. FastAPI App

main = FastAPI(title="Paysim Fraud Detection API", version="2.0.0")


# 4. Load model, scaler, threshold, SHAP

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_")
model = joblib.load(os.path.join(BASE_DIR, "fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
with open(os.path.join(BASE_DIR, "threshold.json")) as f:
    THRESHOLD = json.load(f)["threshold"]
n_features = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else 5
print(f"Model loaded. Threshold: {THRESHOLD} | Features: {n_features}")

try:
    explainer = shap.TreeExplainer(model)
    print("SHAP explainer loaded.")
except Exception as e:
    print(f"SHAP explainer error: {e}")
    explainer = None


# 5. Redis Connection

r = None
try:
    r = redis.Redis(**REDIS_CONFIG)
    r.ping()
    print("Connected to Redis.")
except Exception as e:
    print(f"Redis unavailable: {e}")

#  Database Schema Migration

@main.on_event("startup")
def setup_database():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        # Add columns if they don't exist
        cur.execute("""
            ALTER TABLE predictions 
            ADD COLUMN IF NOT EXISTS oldbalanceorg NUMERIC,
            ADD COLUMN IF NOT EXISTS oldbalancedest NUMERIC,
            ADD COLUMN IF NOT EXISTS step INTEGER,
            ADD COLUMN IF NOT EXISTS transaction_type VARCHAR(20),
            ADD COLUMN IF NOT EXISTS velocity_1m INTEGER,
            ADD COLUMN IF NOT EXISTS avg_amount_1h NUMERIC
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Database schema up to date.")
    except Exception as e:
        print(f"DB schema update failed: {e}")

# Pydantic Payload
class TransactionPayload(BaseModel):
    transaction_id: str
    amount: float
    oldbalanceOrg: float
    oldbalanceDest: float
    step: int
    transaction_type: str
    nameOrig: str

# 8. POST /v1/predict

@main.post("/v1/predict")
async def predict_transaction(payload: TransactionPayload):
    print("PREDICT ENDPOINT EXECUTING (VERSION 2)") 
    if model is None or scaler is None:
        raise HTTPException(500, "Model not loaded.")

    # Redis context
    velocity_1m, avg_amount_1h = 0, 0.0
    if r:
        ctx = r.get(f"user:{payload.nameOrig}")
        if ctx:
            try:
                d = json.loads(ctx)
                velocity_1m = d.get("velocity_1m", 0)
                avg_amount_1h = d.get("avg_amount_1h", 0.0)
            except:
                pass

    hour_of_day = payload.step % 24
    type_is_transfer = 1 if payload.transaction_type.upper() == "TRANSFER" else 0

    if n_features == 5:
        X = [[payload.amount, payload.oldbalanceOrg, payload.oldbalanceDest, hour_of_day, type_is_transfer]]
    else:
        X = [[payload.amount, payload.oldbalanceOrg, payload.oldbalanceDest, hour_of_day, type_is_transfer,
              velocity_1m, avg_amount_1h]]

    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    action = "BLOCKED" if prob >= THRESHOLD else "APPROVED"

    # Save to DB (include velocity_1m and avg_amount_1h)
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print(f"Storing: velocity_1m={velocity_1m}, avg_amount_1h={avg_amount_1h}")
        cur.execute("""
    INSERT INTO predictions (
        transaction_id, fraud_probability, account_number,
        amount, action, threshold_used,
        oldbalanceorg, oldbalancedest, step, transaction_type,
        velocity_1m, avg_amount_1h
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""", (
    payload.transaction_id, round(prob, 4), payload.nameOrig,
    payload.amount, action, THRESHOLD,
    payload.oldbalanceOrg, payload.oldbalanceDest,
    payload.step, payload.transaction_type,
    velocity_1m, avg_amount_1h
))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB insert failed: {e}")

    # Send to Kafka
    result = {
        "transaction_id": payload.transaction_id,
        "fraud_probability": round(prob, 4),
        "account_number": payload.nameOrig,
        "amount": payload.amount,
        "action": action,
        "threshold_used": THRESHOLD
    }
    try:
        kafka_producer.produce('fraud_results', value=json.dumps(result).encode('utf-8'), callback=delivery_report)
        kafka_producer.flush()
    except Exception as e:
        print(f"Kafka produce failed: {e}")

    return {
        "transaction_id": payload.transaction_id,
        "action": action,
        "fraud_probability": round(prob, 4),
        "threshold_used": THRESHOLD,
        "context_features": {
            "velocity_1m": velocity_1m,
            "avg_amount_1h": avg_amount_1h,
            "used_in_prediction": n_features == 7
        }
    }

# 9. Health

@main.get("/v1/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": THRESHOLD,
        "redis_connected": r is not None,
        "features_expected": n_features,
        "shap_loaded": explainer is not None
    }

# 10. SHAP helper

def get_explanation(amount, oldbalanceorg, oldbalancedest, step, transaction_type, action,
                    velocity_1m=0, avg_amount_1h=0.0):
    """Return a plain‑English reason using SHAP contributions and raw values."""
    if explainer is None or scaler is None:
        return "Explanation unavailable."

    try:
        amount = float(amount or 0.0)
        oldbalanceorg = float(oldbalanceorg or 0.0)
        oldbalancedest = float(oldbalancedest or 0.0)
        step = int(step or 0)
        transaction_type = transaction_type or "UNKNOWN"

        hour_of_day = step % 24
        type_is_transfer = 1 if transaction_type.upper() == "TRANSFER" else 0

        if n_features == 5:
            features = [amount, oldbalanceorg, oldbalancedest, hour_of_day, type_is_transfer]
            names = ['amount', 'sender balance', 'recipient balance', 'time of day', 'transfer type']
            raw_values = [amount, oldbalanceorg, oldbalancedest, hour_of_day, type_is_transfer]
        else:
            features = [amount, oldbalanceorg, oldbalancedest, hour_of_day, type_is_transfer,
                        velocity_1m, avg_amount_1h]
            names = ['amount', 'sender balance', 'recipient balance', 'time of day', 'transfer type',
                     'recent transaction speed', 'average amount (last hour)']
            raw_values = [amount, oldbalanceorg, oldbalancedest, hour_of_day, type_is_transfer,
                          velocity_1m, avg_amount_1h]

        X_scaled = scaler.transform([features])
        shap_values = explainer.shap_values(X_scaled)
        contrib = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        pairs = list(zip(names, contrib, raw_values))
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

        if action == "BLOCKED":
            drivers = [(name, c, val) for name, c, val in pairs_sorted if c > 0]
            if not drivers:
                return "Blocked because the overall risk score was high, but no single factor stood out."
            desc = []
            for name, c, val in drivers[:2]:
                if name == 'amount':
                    desc.append(f"the amount was ₦{val:,.2f} (higher than normal)")
                elif name == 'sender balance':
                    desc.append(f"the sender balance was ₦{val:,.2f} (suspiciously low)")
                elif name == 'recipient balance':
                    desc.append(f"the recipient balance was ₦{val:,.2f} (unusual)")
                elif name == 'time of day':
                    desc.append(f"the transaction time (hour {val}) was unusual")
                elif name == 'transfer type':
                    desc.append(f"the transaction was a {transaction_type.lower()} (often risky)")
                elif name == 'recent transaction speed':
                    desc.append(f"there were {val} transactions in the last minute (very high)")
                elif name == 'average amount (last hour)':
                    desc.append(f"the average amount recently was ₦{val:,.2f} (higher than typical)")
                else:
                    desc.append(f"{name} was unusual")
            return "Blocked because " + ", and ".join(desc) + "."

        else:  # APPROVED
            safeguards = [(name, c, val) for name, c, val in pairs_sorted if c < 0]
            if not safeguards:
                return "Approved – no strong fraud indicators found."
            desc = []
            for name, c, val in safeguards[:2]:
                if name == 'amount':
                    desc.append(f"the amount was ₦{val:,.2f} (low, which is normal)")
                elif name == 'sender balance':
                    desc.append(f"the sender had ₦{val:,.2f} in balance (sufficient)")
                elif name == 'recipient balance':
                    desc.append(f"the recipient balance was ₦{val:,.2f} (normal)")
                elif name == 'time of day':
                    desc.append(f"the transaction time (hour {val}) was typical")
                elif name == 'transfer type':
                    desc.append(f"the transfer type ({transaction_type.lower()}) is common")
                elif name == 'recent transaction speed':
                    desc.append(f"there were only {val} transactions in the last minute (low activity)")
                elif name == 'average amount (last hour)':
                    desc.append(f"the average amount recently was ₦{val:,.2f} (consistent with history)")
                else:
                    desc.append(f"{name} was within normal range")
            return "Approved because " + ", and ".join(desc) + "."

    except Exception as e:
        return f"Explanation error: {str(e)}"

# 11. GET endpoints (with behavioural features)


@main.get("/v1/account/{nameOrig}/transactions")
async def get_account_transactions(nameOrig: str, limit: int = 10):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT transaction_id, fraud_probability, amount, action, created_at,
                   oldbalanceorg, oldbalancedest, step, transaction_type,
                   velocity_1m, avg_amount_1h
            FROM predictions
            WHERE account_number = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (nameOrig, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        if not rows:
            raise HTTPException(404, f"No transactions found for account {nameOrig}")

        response = []
        for row in rows:
            reason = get_explanation(
                row['amount'],
                row['oldbalanceorg'],
                row['oldbalancedest'],
                row['step'],
                row['transaction_type'],
                row['action'],
                velocity_1m=row.get('velocity_1m', 0),
                avg_amount_1h=row.get('avg_amount_1h', 0.0)
            )
            response.append({
                "transaction_id": row['transaction_id'],
                "amount": row['amount'],
                "action": row['action'],
                "fraud_probability": row['fraud_probability'],
                "created_at": row['created_at'],
                "reason": reason
            })
        return {"account": nameOrig, "transactions": response}
    except Exception as e:
        raise HTTPException(500, str(e))

@main.get("/v1/account/{nameOrig}/last")
async def get_last_transaction(nameOrig: str):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT transaction_id, fraud_probability, amount, action, created_at,
                   oldbalanceorg, oldbalancedest, step, transaction_type,
                   velocity_1m, avg_amount_1h
            FROM predictions
            WHERE account_number = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (nameOrig,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            raise HTTPException(404, f"No transactions found for account {nameOrig}")
        reason = get_explanation(
            row['amount'],
            row['oldbalanceorg'],
            row['oldbalancedest'],
            row['step'],
            row['transaction_type'],
            row['action'],
            velocity_1m=row.get('velocity_1m', 0),
            avg_amount_1h=row.get('avg_amount_1h', 0.0)
        )
        return {
            "account": nameOrig,
            "last_transaction": {
                "transaction_id": row['transaction_id'],
                "amount": row['amount'],
                "action": row['action'],
                "fraud_probability": row['fraud_probability'],
                "created_at": row['created_at'],
                "reason": reason
            }
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@main.get("/v1/account/{nameOrig}/stats")
async def get_account_stats(nameOrig: str):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN action = 'BLOCKED' THEN 1 ELSE 0 END) AS blocked,
                   COALESCE(SUM(amount),0) AS total_amount,
                   COALESCE(AVG(amount),0) AS avg_amount
            FROM predictions
            WHERE account_number = %s
        """, (nameOrig,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row or row[0] == 0:
            raise HTTPException(404, f"No transactions for account {nameOrig}")
        return {
            "account": nameOrig,
            "total_transactions": row[0],
            "blocked_count": row[1],
            "total_amount": float(row[2]),
            "avg_amount": float(row[3])
        }
    except Exception as e:
        raise HTTPException(500, str(e))