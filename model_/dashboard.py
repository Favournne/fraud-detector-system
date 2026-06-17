# dashboard.py
from datetime import datetime
import json
import os
import random
import sqlite3
import time

from faker import Faker
from kafka import KafkaProducer
import pandas as pd
import streamlit as st

# =====================================================================
# CONFIGURATION
# =====================================================================
DB_PATH = "fraud_events.db"
TOPIC_NAME = "raw_transactions"
REFRESH_SECONDS = 2

fake = Faker(["en_NG"])

st.set_page_config(
    page_title="FinCrime Streaming Observer", page_icon="🛡️", layout="wide"
)

# --- SAFE DB DIAGNOSTIC ---
try:
    conn = sqlite3.connect(DB_PATH)
    db_path = os.path.abspath(DB_PATH)
    print(f"\nDashboard is reading DB file from: {db_path}")

    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(events);")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Actual columns in SQLite table 'events': {columns}\n")
    conn.close()
except Exception as e:
    print(f"Diagnostic check skipped or failed: {e}")

st.markdown(
    """
    <style>
    .stApp { background-color: #FFFFFF; color: #1F2937; }
    h1, h2, h3, h4 { color: #1E3A8A !important; font-family: 'Inter', sans-serif; font-weight: bold; }
    .stMetric {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #7C3AED;
    }
    div[data-testid="stMetricValue"] { color: #1E3A8A; font-family: monospace; font-weight: bold; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("FinCrime Real-Time Detection Monitor")
st.markdown("### Production Streaming Telemetry & Model Evaluation Ledger")
st.markdown("---")


# =====================================================================
# KAFKA PRODUCER (for attack injection buttons)
# =====================================================================
@st.cache_resource
def get_producer():
    try:
        return KafkaProducer(
            bootstrap_servers=["localhost:9092"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    except Exception as e:
        st.error(f"Could not connect to Kafka: {e}")
        return None


producer = get_producer()


# =====================================================================
# DATA ACCESS — reads model verdicts logged by kafka_consumer.py
# =====================================================================
def load_events(limit=15):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            f"SELECT * FROM events ORDER BY id DESC LIMIT {limit}", conn
        )
        conn.close()

        # Dynamic fallback: If schema is outdated or empty, ensure the column exists
        if not df.empty and "anomaly_score" not in df.columns:
            df["anomaly_score"] = 0.0
        if "account_id" not in df.columns and "account_number" in df.columns:
                df = df.rename(columns={"account_number": "account_id"})

        return df
    except Exception:
        return pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "transaction_id",
                "account_id",
                "amount",
                "channel",
                "location",
                "is_anomaly",
                "anomaly_score",
                "action_required",
            ]
        )


def load_totals():
    try:
        conn = sqlite3.connect(DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        flagged = conn.execute(
            "SELECT COUNT(*) FROM events WHERE is_anomaly = 1"
        ).fetchone()[0]
        saved = (
            conn.execute(
                "SELECT SUM(amount) FROM events WHERE is_anomaly = 1"
            ).fetchone()[0]
            or 0.0
        )
        conn.close()
        return total, flagged, saved
    except Exception:
        return 0, 0, 0.0


# =====================================================================
# ATTACK PAYLOAD BUILDERS
# =====================================================================
def base_payload(**overrides):
    payload = {
        "TransactionID": f"TXN-{random.randint(1000000000, 9999999999)}",
        "AccountID": f"ACC_{random.randint(10000, 99999)}",
        "TransactionAmount": 5000.0,
        "TransactionDate": datetime.now().strftime("%Y-%m-%d"),
        "TransactionType": "Debit",
        "Location": "Lagos",
        "DeviceID": f"DEV-{random.randint(100000, 999999)}",
        "IP Address": fake.ipv4(),
        "MerchantID": f"MERCH_{random.randint(100, 999)}",
        "Channel": "Mobile App",
        "TransactionDuration": round(random.uniform(10.0, 180.0), 1),
        "LoginAttempts": random.randint(4, 5),
        "AccountBalance": round(random.uniform(5000, 500000), 2),
    }
    payload.update(overrides)
    return payload


def ussd_late_night_drain():
    return base_payload(
        Channel="USSD Code (*737#)",
        TransactionAmount=round(random.uniform(120000, 480000), 2),
        TransactionType="Debit",
        DeviceID="NaN",
    )


def pos_behavioral_anomaly():
    return base_payload(
        Channel="POS",
        TransactionAmount=round(random.uniform(200000, 600000), 2),
        Location=random.choice(["Kano", "Enugu", "Edo", "Ondo"]),
        DeviceID="NaN",
    )


def device_pooling_attack():
    rogue_devices = [f"DEV-ROGUE{str(i).zfill(2)}" for i in range(15)]
    return base_payload(
        Channel="Mobile App",
        TransactionAmount=round(random.uniform(120000, 480000), 2),
        DeviceID=random.choice(rogue_devices),
        TransactionType="Debit",
    )


def send_attack(payload_fn, label):
    if producer is None:
        st.error("Kafka producer not connected — is Kafka running?")
        return
    payload = payload_fn()
    producer.send(TOPIC_NAME, value=payload)
    producer.flush()
    st.success(
        f"Injected → {label} | {payload['TransactionID']} | "
        f"₦{payload['TransactionAmount']:,.2f} via {payload['Channel']}"
    )


# =====================================================================
# KPI ROW
# =====================================================================
total_monitored, total_flagged, total_saved = load_totals()
deflection_rate = (
    (total_flagged / total_monitored * 100) if total_monitored > 0 else 0.0
)

m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Total Transactions Audited", f"{total_monitored:,}")
m_col2.metric(
    "Model Deflections",
    f"{total_flagged:,} Threats",
    f"{deflection_rate:.1f}% deflection rate",
)
m_col3.metric("Preserved Ecosystem Capital", f"₦{total_saved:,.2f}")

st.markdown("---")

# =====================================================================
# ATTACK INJECTION PANEL
# =====================================================================
st.subheader("Inject a Live Fraud Pattern")
st.caption(
    "Fires a crafted transaction payload directly into the Kafka stream, "
    "matching one of the system's designed fraud scenarios. "
    "Watch the ticker below react within seconds."
)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("USSD Late-Night Drain", use_container_width=True):
        send_attack(ussd_late_night_drain, "USSD Late-Night Drain")
with col2:
    if st.button("POS Behavioral Anomaly", use_container_width=True):
        send_attack(pos_behavioral_anomaly, "POS Behavioral Anomaly")
with col3:
    if st.button("Device Pooling Attack", use_container_width=True):
        send_attack(device_pooling_attack, "Device Pooling Attack")

st.markdown("---")

# =====================================================================
# LIVE TICKER
# =====================================================================
st.subheader("Real-Time Production Transaction Stream")

# df is defined here correctly now
df = load_events(limit=15)

if df.empty:
    st.info(
        "No events logged yet. "
        "Make sure kafka_consumer.py is running and the simulator is streaming."
    )
else:
    display_df = df[
        [
            "timestamp",
            "transaction_id",
            "account_id",
            "amount",
            "channel",
            "location",
            "anomaly_score",
            "action_required",
            "is_anomaly",
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "timestamp": "Timestamp",
            "transaction_id": "Transaction ID",
            "account_id": "Account ID",
            "amount": "Amount",
            "channel": "Channel",
            "location": "Location",
            "anomaly_score": "Anomaly Score",
            "action_required": "Action",
            "is_anomaly": "_flag",
        }
    )

    display_df["Status"] = display_df["_flag"].map(
        {1: "BLOCKED", 0: "APPROVED"}
    )
    display_df = display_df.drop(columns=["_flag"])

    def highlight_fraud_rows(row):
        if "BLOCKED" in str(row["Status"]):
            return [
                "background-color: #FEE2E2; color: #991B1B; font-weight: bold;"
            ] * len(row)
        return ["background-color: #FFFFFF; color: #1F2937;"] * len(row)

    styled_df = display_df.style.apply(highlight_fraud_rows, axis=1).format(
        {"Amount": "₦{:,.2f}", "Anomaly Score": "{:.4f}"}
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.caption(
    f"Last updated: {datetime.now().strftime('%H:%M:%S')} · auto-refreshes every {REFRESH_SECONDS}s"
)

# =====================================================================
# AUTO-REFRESH
# =====================================================================
time.sleep(REFRESH_SECONDS)
st.rerun()