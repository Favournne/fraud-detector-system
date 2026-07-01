# producer_live.py
import json
import time
import random
import numpy as np
import pandas as pd
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# 1. Load the ORIGINAL dataset to learn distributions
df = pd.read_csv(r"../data_/paysim.csv")  # adjust path if needed

# 2. Filter to only the types your model accepts (TRANSFER/CASH_OUT)
df_sample = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]

# 3. Learn statistical profiles
legit = df_sample[df_sample['isFraud'] == 0]
fraud = df_sample[df_sample['isFraud'] == 1]

def generate_transaction():
    # Decide if this transaction will be fraud (matches ~0.3% rate)
    is_fraud = random.random() < 0.003  # 0.3% fraud rate
    
    if is_fraud:
        base = fraud.sample(1).iloc[0]
    else:
        base = legit.sample(1).iloc[0]
    
    # Add random noise to amount and balances (simulate live variation)
    amount_noise = np.random.normal(1.0, 0.05)
    new_amount = max(1, base['amount'] * amount_noise)
    
    # Randomise step (hour) to simulate time progression
    new_step = random.randint(0, 23)
    
    # Build the transaction WITH REAL USER IDS
    return {
        "transaction_id": f"LIVE-{int(time.time())}-{random.randint(100,999)}",
        "nameOrig": base['nameOrig'],                 # <-- REAL origin account
        "nameDest": base['nameDest'],                 # <-- REAL destination
        "amount": round(new_amount, 2),
        "oldbalanceOrg": round(base['oldbalanceOrg'] * np.random.normal(1.0, 0.02), 2),
        "oldbalanceDest": round(base['oldbalanceDest'] * np.random.normal(1.0, 0.02), 2),
        "step": new_step,
        "transaction_type": "TRANSFER" if random.random() > 0.5 else "CASH_OUT",
        # Keep these if you want for display, but they are NOT used for context
        "customer_name": f"Customer_{random.randint(1, 1000)}"   # optional
    }

# 4. Stream forever
print("🚀 Starting live transaction generator with real nameOrig...")
while True:
    tx = generate_transaction()
    producer.send('transactions', value=tx)
    print(f"Sent: {tx['transaction_id']} | User: {tx['nameOrig']} | Amount: {tx['amount']}")
    time.sleep(random.uniform(0.5, 2.0))  # simulate real‑time gap