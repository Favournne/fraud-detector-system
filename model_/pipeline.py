import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from features import FraudFeatureEngineer

class FraudDataPipeline:
    """
    Upgraded Production Data Pipeline.
    Handles data ingestion, chronological splitting, and behavioral feature engineering.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None

    def load_and_inspect(self):
        """Safely load and print structural dimensions of the raw log data."""
        print(f" Ingesting raw dataset from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)

        # Enforce chronological sequence immediately to prevent downstream data leakage
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(by='timestamp').reset_index(drop=True)

        print("\n--- DATA PROFILE ---")
        print(f"Total Transactions: {self.df.shape[0]}")
        #print(f"Total Features:     {self.df.shape[1]}")
        print(f"📋 Global Time Order Enforced: {self.df['timestamp'].is_monotonic_increasing}")
        return self.df
    

    def engineer_behavioral_features(self):
        """
        Advanced V2.0 Behavioral Engine.
        Engineers structural network features cleanly using numeric conversions
        to bypass internal Pandas rolling-window type constraints.
        """
        print("\n🧠 Engineering Robust Network Behavioral Profiles...")
        
        # 1. Handle missing values uniquely
        self.df['clean_device_id'] = self.df['device_id'].fillna(self.df['account_number'].astype(str) + "_unk_dev")
        self.df['destination_bank'] = self.df['destination_bank'].fillna('internal')
        
        # FIX: Convert categorical columns into temporary integers so Pandas rolling can process them
        self.df['bank_id_encoded'] = self.df['destination_bank'].astype('category').cat.codes
        self.df['account_id_encoded'] = self.df['account_number'].astype('category').cat.codes
        
        # Create temporary time-indexed dataframe for rolling calculations
        temp_df = self.df.set_index('timestamp')
        
        # --- FEATURE 1: Account Velocity (1-Hour) ---
        print(" -> Computing account velocity windows (1-Hour)...")
        user_counts = temp_df.groupby('account_number')['amount'].rolling('1h', closed='left').count().reset_index(level=0, drop=True).fillna(0)
        
        # --- FEATURE 2: Device Velocity (1-Hour) ---
        print(" -> Computing device footprint tracking (1-Hour)...")
        device_counts = temp_df.groupby('clean_device_id')['amount'].rolling('1h', closed='left').count().reset_index(level=0, drop=True).fillna(0)
        
        # --- FEATURE 3: Value Deviation (7-Day) ---
        print(" -> Calculating 7-Day value deviation ratios...")
        user_7d_avg = temp_df.groupby('account_number')['amount'].rolling('7d', closed='left').mean().reset_index(level=0, drop=True)
        
        # --- FEATURE 4: Destination Bank Diversity (1-Hour out-degree) ---
        print(" -> Detecting multi-bank layering velocity (Out-Degree Cardinality)...")
        # Now processing numerical bank_id_encoded arrays instead of raw strings!
        bank_diversity = (
            temp_df.groupby('account_number')['bank_id_encoded']
            .rolling('1h', closed='left')
            .agg(lambda x: len(set(x)))
            .reset_index(level=0, drop=True)
            .fillna(1)
        )
        
        # --- FEATURE 5: Device Pooling Workstations (24-Hour) ---
        print(" -> Flagging syndicated device pooling workstations (24-Hour User Count)...")
        # Now processing numerical account_id_encoded arrays instead of raw strings!
        device_pooling = (
            temp_df.groupby('clean_device_id')['account_id_encoded']
            .rolling('24h', closed='left')
            .agg(lambda x: len(set(x)))
            .reset_index(level=0, drop=True)
            .fillna(1)
        )
        
        # Map calculated arrays safely back to the master dataframe
        self.df['user_tx_count_1h'] = user_counts.values
        self.df['device_tx_count_1h'] = device_counts.values
        self.df['unique_dest_banks_1h'] = bank_diversity.values
        self.df['accounts_per_device_24h'] = device_pooling.values
        
        self.df['amount_vs_avg_7d'] = self.df['amount'] / user_7d_avg.values
        self.df['amount_vs_avg_7d'] = self.df['amount_vs_avg_7d'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
        
        # Clean up temporary structural features
        self.df.drop(columns=['clean_device_id', 'bank_id_encoded', 'account_id_encoded'], inplace=True)
        
        print("✅ Production-grade network profiling complete!")
        return self.df
        
        
        
    def chronological_split(self, test_ratio: float = 0.20):
        """Splits the dataset using a strict time boundary."""
        print(f"\nExecuting chronological split (Test Ratio: {test_ratio})...")
        split_idx = int(len(self.df) * (1 - test_ratio))
        
        train_df = self.df.iloc[:split_idx].copy()
        test_df = self.df.iloc[split_idx:].copy()
        
        return train_df, test_df
    
