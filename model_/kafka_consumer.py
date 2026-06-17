from datetime import datetime
import json
import sqlite3
from kafka import KafkaConsumer
import requests

# --- Configuration & Initialization ---
API_URL = "http://127.0.0.1:8000/v1/predict"
TOPIC_NAME = "raw_transactions"
DB_PATH = "fraud_events.db"


def init_db():
    """Initializes the SQLite database schema if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            transaction_id TEXT,
            account_id TEXT,
            amount REAL,
            channel TEXT,
            location TEXT,
            is_anomaly INTEGER,
            action_required TEXT
        )
    """
    )
    conn.commit()
    conn.close()


# Initialize database and establish connection
init_db()
db_conn = sqlite3.connect(DB_PATH)

print(
    f"📡 Kafka Risk Evaluation Consumer Active. Listening on topic: '{TOPIC_NAME}'..."
)

# Initialize Kafka Consumer
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

        # Payload preparation
        api_payload = {
            "transaction_id": str(tx_id),
            "account_number": str(tx.get("AccountID")).replace("ACC_", ""),  # Extracts raw numbers
            "customer_name": str(tx.get("CustomerName", "Unknown Customer")),
            "device_id": str(tx.get("DeviceID")),
            "amount": float(tx.get("TransactionAmount", 0.0)),
            "channel": str(tx.get("Channel")),
            "transaction_type": str(tx.get("TransactionType")),
            "destination_bank": str(tx.get("MerchantID") ),  # Maps placeholder to destination bank tracking
            "narration": "Live Streaming Feed Event",
            "location_state": str(tx.get("Location")),
        }

        action = "UNKNOWN" 
        anomaly_score = 0.0


        try:
            # Query the Decision Engine API
            response = requests.post(API_URL, json=api_payload)
            if response.status_code == 200:
                score_metadata = response.json()

                # Extract and sanitize the decision engine's action string
                action = str(score_metadata.get("action", "")).strip().upper()

                anomaly_score = float(
                    score_metadata.get(
                        "anomaly_score", score_metadata.get("score", 0.0)
                    )

                )

                if action == "BLOCKED":
                    print(
                        f"ALERT! Anomaly Tracked on {tx_id} | ACTION: {action} | Msg: {score_metadata.get('public_message')}"
                    )
                elif action == "APPROVED":
                    print(f"Approved: {tx_id} | Profile Verified Normal.")
                else:
                    print(f"Unexpected response shape for {tx_id}: {score_metadata}")
            else:
                print(
                    f"API Error processing {tx_id}: Status {response.status_code}"
                )

        except requests.exceptions.ConnectionError:
            print(
                "Connection Refused! Is your FastAPI server process running on port 8000?"
            )
            action = "API_ERROR"

        # Log event to database *inside* the loop so every processed message is captured
        try:
            db_conn.execute(
                """INSERT INTO events 
                   (timestamp, transaction_id, account_id, amount, channel, location, is_anomaly, action_required)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    tx_id,
                    api_payload["account_number"],
                    api_payload["amount"],
                    api_payload["channel"],
                    api_payload["location_state"],
                    1 if action == "BLOCKED" else 0,
                    anomaly_score,
                    action,
                ),
            )
            db_conn.commit()
        except sqlite3.Error as db_err:
            print(f"Database insertion failed for transaction {tx_id}: {db_err}")

except KeyboardInterrupt:
    print("\nConsumer processing loop disconnected.")

finally:
    # Safely close the database connection when exiting
    print("Closing database connection...")
    db_conn.close()