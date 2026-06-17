import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("nigerian_bank_transactions_v3.csv")

print("=== CRACKING THE SYNTHETIC RULES ===")

# 1. Check direct correlations with categorical features
for col in ['channel', 'transaction_type', 'location_state', 'narration']:
    print(f"\n• Fraud Rate by [{col}]:")
    print(df.groupby(col)['is_fraud'].mean().sort_values(ascending=False).head(5))

# 2. Check numeric distributions
print("\n• Average Amount by Class:")
print(df.groupby('is_fraud')['amount'].describe())

# 3. Check time signatures
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
print("\n• Fraud Rate by Hour of Day:")
print(df.groupby('hour')['is_fraud'].mean().sort_values(ascending=False).head(5))