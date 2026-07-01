import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load training split to avoid tampering with validation/test sets
data_path = os.path.join("splits", "train.csv")
if not os.path.exists(data_path):
    # Fallback to current dir if running from inside data_
    data_path = "train.csv"

print(f"Loading {data_path} for an empirical audit...")
df = pd.read_csv(data_path)

print("\n=== 1. CARDINALITY AUDIT ===")
for col in ['nameOrig', 'nameDest']:
    unique_count = df[col].nunique()
    print(f"Column '{col}' has {unique_count:,} unique values out of {len(df):,} total rows.")

print("\n=== 2. UPSTREAM RULE LEAKAGE CHECK ('isFlaggedFraud') ===")
flagged_vs_actual = pd.crosstab(df['isFlaggedFraud'], df['isFraud'])
print(flagged_vs_actual)

print("\n=== 3. SIGNAL VS. NOISE CORRELATION PROFILING ===")
# Create temporary features for comparison
df_audit = df.copy()
df_audit['errorBalanceOrig'] = df_audit['oldbalanceOrg'] - df_audit['amount'] - df_audit['newbalanceOrig']
df_audit['errorBalanceDest'] = df_audit['oldbalanceDest'] + df_audit['amount'] - df_audit['newbalanceDest']
df_audit['hour_of_day'] = df_audit['step'] % 24

# Select numerical columns for tracking linear correlation
numerical_cols = [
    'step', 'hour_of_day', 'amount', 
    'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'errorBalanceDest',
    'isFraud'
]

matrix = df_audit[numerical_cols].corr()
print("\nCorrelation with 'isFraud' target variable (Sorted):")
print(matrix['isFraud'].sort_values(ascending=False))

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title("Empirical Feature Correlation & Leakage Audit Matrix")
plt.tight_layout()
plt.show()