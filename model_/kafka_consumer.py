from datetime import datetime
import json
import sqlite3
import requests
from kafka import KafkaConsumer
import os 



# --- Configuration ---
API_URL = "http://127.0.0.1:8000/v1/predict"
TELEMETRY_URL = "http://127.0.0.1:8000/v1/accounts/{account_number}/fraud-telemetry"
TOPIC_NAME = "raw_transactions"
DB_PATH = r"C:\Users\USER\fraud_project\model_\fraud_events.db"

print(f"!!! CONSUMER WRITING TO: {os.path.abspath(DB_PATH)}")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp               TEXT,
            transaction_id          TEXT,
            account_id              TEXT,
            customer_name           TEXT,
            amount                  REAL,
            channel                 TEXT,
            transaction_type        TEXT,
            location                TEXT,
            device_id               TEXT,
            destination_bank        TEXT,
            is_anomaly              INTEGER,
            fraud_probability       REAL,
            action_required         TEXT,
            user_tx_count_1h        REAL,
            device_tx_count_1h      REAL,
            unique_dest_banks_1h    REAL,
            accounts_per_device_24h REAL,
            amount_vs_avg_7d        REAL
        )
    """)
    conn.commit()
    conn.close()


def fetch_behavioral_signals(account_number: str) -> dict:
    """Pulls behavioral signals from FastAPI telemetry endpoint for BLOCKED accounts."""
    empty = {
        "fraud_probability": 0.0,
        "user_tx_count_1h": 0.0,
        "device_tx_count_1h": 0.0,
        "unique_dest_banks_1h": 0.0,
        "accounts_per_device_24h": 0.0,
        "amount_vs_avg_7d": 0.0,
    }
    try:
        resp = requests.get(
            TELEMETRY_URL.format(account_number=account_number), timeout=2
        )
        if resp.status_code == 200:
            telemetry = resp.json().get("telemetry", {})
            signals = telemetry.get("live_behavioral_signals", {})
            return {
                "fraud_probability": float(telemetry.get("fraud_probability_score", 0.0)),
                "user_tx_count_1h": float(signals.get("user_tx_count_last_hour", 0.0)),
                "device_tx_count_1h": float(signals.get("device_tx_count_last_hour", 0.0)),
                "unique_dest_banks_1h": float(signals.get("distinct_banks_targeted_1h", 0.0)),
                "accounts_per_device_24h": float(signals.get("accounts_switched_on_device_24h", 0.0)),
                "amount_vs_avg_7d": float(signals.get("amount_deviation_multiplier_7d", 0.0)),
            }
    except Exception:
        pass
    return empty


# --- Initialization ---
# Update it to look like this:
init_db()
db_conn = sqlite3.connect(DB_PATH, isolation_level=None) 

print(f"📡 Kafka Risk Evaluation Consumer Active. Listening on topic: '{TOPIC_NAME}'...")

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=["localhost:9092"],
    auto_offset_reset="latest",
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
)

# --- Main Event Processing Loop ---
try:
    for message in consumer:
        tx = message.value
        tx_id = tx.get("TransactionID")

        api_payload = {
            "transaction_id": str(tx_id),
            "account_number": str(tx.get("AccountID")).replace("ACC_", ""),
            "customer_name": str(tx.get("CustomerName", "Unknown Customer")),
            "device_id": str(tx.get("DeviceID")),
            "amount": float(tx.get("TransactionAmount", 0.0)),
            "channel": str(tx.get("Channel")),
            "transaction_type": str(tx.get("TransactionType")),
            "destination_bank": str(tx.get("MerchantID")),
            "narration": "Live Streaming Feed Event",
            "location_state": str(tx.get("Location")),
        }

        # Defaults
        action = "UNKNOWN"
        fraud_probability = 0.0
        signals = {
            "user_tx_count_1h": 0.0,
            "device_tx_count_1h": 0.0,
            "unique_dest_banks_1h": 0.0,
            "accounts_per_device_24h": 0.0,
            "amount_vs_avg_7d": 0.0,
        }

        try:
            response = requests.post(API_URL, json=api_payload)
            if response.status_code == 200:
                score_metadata = response.json()
                action = str(score_metadata.get("action", "")).strip().upper()

                if action == "BLOCKED":
                    print(f"🚨 ALERT! {tx_id} | BLOCKED | {score_metadata.get('public_message')}")
                    enriched = fetch_behavioral_signals(api_payload["account_number"])
                    fraud_probability = enriched.pop("fraud_probability")
                    signals = enriched
                elif action == "APPROVED":
                    print(f"✅ Approved: {tx_id} | Profile Verified Normal.")
                else:
                    print(f"⚠️ Unexpected response for {tx_id}: {score_metadata}")
            else:
                print(f"API Error processing {tx_id}: Status {response.status_code}")
                action = "API_ERROR"

        except requests.exceptions.ConnectionError:
            print("❌ Connection Refused! Is FastAPI running on port 8000?")
            action = "API_ERROR"

        # --- INSERT — columns and values match exactly (18 each) ---
        try:
            db_conn.execute(
                """INSERT INTO events (
                    timestamp, transaction_id, account_id, customer_name,
                    amount, channel, transaction_type, location,
                    device_id, destination_bank,
                    is_anomaly, fraud_probability, action_required,
                    user_tx_count_1h, device_tx_count_1h, unique_dest_banks_1h,
                    accounts_per_device_24h, amount_vs_avg_7d
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 1  timestamp
                    tx_id,                                          # 2  transaction_id
                    api_payload["account_number"],                  # 3  account_id
                    api_payload["customer_name"],                   # 4  customer_name
                    api_payload["amount"],                          # 5  amount
                    api_payload["channel"],                         # 6  channel
                    api_payload["transaction_type"],                # 7  transaction_type
                    api_payload["location_state"],                  # 8  location
                    api_payload["device_id"],                       # 9  device_id
                    api_payload["destination_bank"],                # 10 destination_bank
                    1 if action == "BLOCKED" else 0,               # 11 is_anomaly
                    fraud_probability,                              # 12 fraud_probability
                    action,                                         # 13 action_required
                    signals["user_tx_count_1h"],                   # 14
                    signals["device_tx_count_1h"],                 # 15
                    signals["unique_dest_banks_1h"],               # 16
                    signals["accounts_per_device_24h"],            # 17
                    signals["amount_vs_avg_7d"],                   # 18
                ),
            )
            db_conn.commit()
        except sqlite3.Error as db_err:
            print(f"❌ DB insertion failed for {tx_id}: {db_err}")

except KeyboardInterrupt:
    print("\n🛑 Consumer processing loop disconnected.")
finally:
    print("Closing database connection...")
    db_conn.close()