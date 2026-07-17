# dashboard.py
import os
from dotenv import load_dotenv

# Load .env from project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)

import json
import time
import pandas as pd
import streamlit as st
import psycopg2
from confluent_kafka import Consumer, KafkaError


DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "fraud_stream_db"),
    "user": os.getenv("POSTGRES_USER", "fraud_user"),
    "password": os.environ["POSTGRES_PASSWORD"]   # <-- no fallback
}

st.set_page_config(page_title="Fraud Monitor", layout="wide")

# ---------- Sidebar navigation ----------
st.sidebar.title("Navigation")
st.sidebar.markdown("[Live Monitor](http://localhost:8501)")
st.sidebar.markdown("[Account Investigation](http://localhost:8502)")
st.sidebar.divider()
st.sidebar.caption("Data from Kafka & PostgreSQL")


st.title("Live Fraud Detection Dashboard")
st.caption("Streaming ALL transactions from Kafka (fraud_results topic)")


def fetch_kpis():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions;")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM predictions WHERE action = 'BLOCKED';")
        frauds = cur.fetchone()[0]
        cur.execute("SELECT COALESCE(SUM(amount), 0) FROM predictions WHERE action = 'BLOCKED';")
        saved = cur.fetchone()[0]
        cur.close()
        conn.close()
        fraud_rate = (frauds / total * 100) if total > 0 else 0.0
        return total, frauds, saved, fraud_rate
    except Exception as e:
        return 0, 0, 0.0, 0.0

# ---------- Kafka consumer ----------
@st.cache_resource
def get_consumer():
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'dashboard-group-v3',   # <-- new group to force reading from start
        'auto.offset.reset': 'earliest',#'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
        'session.timeout.ms': 6000,
    }
    return Consumer(conf)

placeholder = st.empty()
transactions = []

# ---------- Table styling ----------
def highlight_blocked(row):
    if row['action'] == 'BLOCKED':
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)

def main_loop():
    consumer = get_consumer()
    consumer.subscribe(['fraud_results'])
    
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            pass
        elif msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print(f"Consumer error: {msg.error()}")
        else:
            try:
                value = json.loads(msg.value().decode('utf-8'))
                transactions.append(value)
            except Exception as e:
                print(f"Decode error: {e}")
        
        with placeholder.container():
            # KPIs
            total, frauds, saved, fraud_rate = fetch_kpis()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", f"{total:,}")
            col2.metric("Frauds Caught", f"{frauds:,}")
            col3.metric("Money Saved", f"₦{saved:,.2f}")
            col4.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            st.markdown("---")
            
            # Transaction table
            if transactions:
                df = pd.DataFrame(transactions)
                display_df = df.copy()
                # Rename for clarity
                if 'account_number' in display_df.columns:
                    display_df = display_df.rename(columns={'account_number': 'nameOrig'})
                
                # Format currency and probability
                currency_cols = ['amount', 'oldbalanceOrg', 'oldbalanceDest']
                for col in currency_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"₦{float(x):,.2f}")
                if 'fraud_probability' in display_df.columns:
                    display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{float(x):.2%}")
                
                # Apply red styling to BLOCKED rows
                styled = display_df.style.apply(highlight_blocked, axis=1)
                st.dataframe(styled, use_container_width=True)
            else:
                st.info("Waiting for transactions...")
        
        time.sleep(2)

if __name__ == "__main__":
    main_loop()


    # streamlit run dashboard.py  # streamlit run dashboard.py
#C1039013014