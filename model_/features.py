import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Production-grade Feature Engineering Engine V2.0.
    Implements leak-proof out-of-fold target encoding with global smoothing,
    preserving behavioral and network graph features across train/test footprints.
    """
    def __init__(self, m_smoothing: float = 100.0):
        self.m_smoothing = m_smoothing
        self.target_encodings = {}
        self.global_mean = 0.0
        self.outlier_threshold = 55779.26
        self.engineered_feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Learns smoothed target mappings securely across the entire training footprint."""
        if y is None:
            raise ValueError("Target 'y' is required to fit the feature engineer.")
            
        temp_df = X.copy()
        temp_df['target'] = y.loc[X.index]
        
        self.global_mean = temp_df['target'].mean()
        categorical_cols = ['channel', 'transaction_type', 'location_state']
        
        for col in categorical_cols:
            if col in temp_df.columns:
                stats = temp_df.groupby(col)['target'].agg(['count', 'mean'])
                smoothed_vals = (stats['count'] * stats['mean'] + self.m_smoothing * self.global_mean) / (stats['count'] + self.m_smoothing)
                self.target_encodings[col] = smoothed_vals.to_dict()
                
        return self

    def _handle_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal worker method to clean structural anomalies."""
        df = df.copy()
        if 'device_id' in df.columns:
            df['device_id'] = df['device_id'].fillna('unknown_device')
        if 'destination_bank' in df.columns:
            df['destination_bank'] = df['destination_bank'].fillna('internal_transaction')
        return df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None, **fit_params) -> pd.DataFrame:
        """Enforces K-Fold Out-of-Fold encoding on Training data."""
        self.fit(X, y)      
        df = self._handle_cleaning(X)
        df['target'] = y.loc[X.index]
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        if 'amount' in df.columns:
            df['is_high_value'] = (df['amount'] > self.outlier_threshold).astype(int)
            df['amount_log'] = np.log1p(df['amount'])
            
        categorical_cols = ['channel', 'transaction_type', 'location_state']
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for col in categorical_cols:
            if col in df.columns:
                encoded_col_name = f"{col}_risk_score"
                df[encoded_col_name] = self.global_mean
                
                for train_idx, val_idx in kf.split(df):
                    fold_train = df.iloc[train_idx]
                    stats = fold_train.groupby(col)['target'].agg(['count', 'mean'])
                    smoothed = (stats['count'] * stats['mean'] + self.m_smoothing * self.global_mean) / (stats['count'] + self.m_smoothing)
                    df.iloc[val_idx, df.columns.get_loc(encoded_col_name)] = df.iloc[val_idx][col].map(smoothed).fillna(self.global_mean)

        # LOCKED 12-FEATURE LAYOUT FOR TRAINING
        model_ready_features = [
            'amount_log', 'is_high_value', 'hour_of_day', 'day_of_week',
            'channel_risk_score', 'transaction_type_risk_score', 'location_state_risk_score',
            'user_tx_count_1h', 'device_tx_count_1h', 'amount_vs_avg_7d',
            'unique_dest_banks_1h', 'accounts_per_device_24h'
        ]
        self.engineered_feature_names = [f for f in model_ready_features if f in df.columns]
        return df[self.engineered_feature_names].copy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies global mappings and safely forwards behavioral context metrics."""
        df = self._handle_cleaning(X)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        if 'amount' in df.columns:
            df['is_high_value'] = (df['amount'] > self.outlier_threshold).astype(int)
            df['amount_log'] = np.log1p(df['amount'])

        for col, encoding_map in self.target_encodings.items():
            encoded_col_name = f"{col}_risk_score"
            if col in df.columns:
                df[encoded_col_name] = df[col].map(encoding_map).fillna(self.global_mean)
            else:
                df[encoded_col_name] = self.global_mean

        # IDENTICAL LOCKED 12-FEATURE LAYOUT FOR EVALUATION/INFERENCE
        model_ready_features = [
            'amount_log', 'is_high_value', 'hour_of_day', 'day_of_week',
            'channel_risk_score', 'transaction_type_risk_score', 'location_state_risk_score',
            'user_tx_count_1h', 'device_tx_count_1h', 'amount_vs_avg_7d',
            'unique_dest_banks_1h', 'accounts_per_device_24h'
        ]
        return df[model_ready_features].copy()