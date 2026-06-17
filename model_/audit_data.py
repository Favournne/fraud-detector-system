import pandas as pd
import numpy as np

def audit_dataset(csv_path):
    print(f"🕵️‍♂️ Initiating Deep Forensic Audit on: {csv_path}\n")
    df = pd.read_csv(csv_path)
    
    print("=== STRUCTURAL INTEGRITY ===")
    print(f"Total Rows: {len(df)}")
    
    # 1. Check for absolute row duplicates
    exact_dup = df.duplicated().sum()
    print(f"Absolute Duplicate Rows (Complete Mirror Images): {exact_dup}")
    
    # 2. Check for Transaction ID integrity
    if 'transaction_id' in df.columns:
        tx_dup = df.duplicated(subset=['transaction_id']).sum()
        print(f"Rows with Duplicate transaction_id keys: {tx_dup}")
    
    print("\n=== THE TIMELINE AUDIT ===")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    is_sorted = df['timestamp'].is_monotonic_increasing
    print(f"Is the raw data naturally sorted by time?: {is_sorted}")
    
    # Check for identical timestamps (Simultaneous transactions)
    same_time_events = df.duplicated(subset=['timestamp'], keep=False).sum()
    print(f"Rows sharing an exact identical timestamp millisecond: {same_time_events}")

    print("\n=== MISSING VALUE & CARDINALITY BREAKDOWN ===")
    nulls = df.isnull().sum()
    for col in df.columns:
        uniques = df[col].nunique()
        print(f"• Column [{col:22}]: Nulls = {nulls[col]:<6} | Unique Values = {uniques}")
        
    print("\n=== TARGET DISTRIBUTION ===")
    if 'is_fraud' in df.columns:
        fraud_counts = df['is_fraud'].value_counts()
        fraud_pct = df['is_fraud'].value_mean() * 100 if hasattr(df['is_fraud'], 'value_mean') else (df['is_fraud'].sum() / len(df)) * 100
        print(f"Legitimate Logs: {df['is_fraud'].value_counts().get(0, 0)}")
        print(f"Fraudulent Logs: {df['is_fraud'].value_counts().get(1, 0)} ({fraud_pct:.4f}%)")

if __name__ == "__main__":
    audit_dataset("nigerian_bank_transactions_v3.csv")