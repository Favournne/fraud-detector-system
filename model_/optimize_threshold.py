import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def run_threshold_optimization(data_path="nigerian_bank_transactions_v3.csv", model_path=r"C:\Users\USER\fraud_project\model_\models\fraud_pipeline_v1.0.pkl"):
    print("📈 Ingesting raw dataset for threshold calibration...")
    df = pd.read_csv(data_path)
    
    # Enforce global chronological order
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print("🧠 Engineering behavioral network profiles locally...")
    # 1. Isolate device profiles cleanly
    df['clean_device_id'] = df['device_id'].fillna(df['account_number'].astype(str) + "_unk_dev")
    df['destination_bank'] = df['destination_bank'].fillna('internal')
    
    # 2. Encode identifiers to integers for Pandas rolling compatibility
    df['bank_id_encoded'] = df['destination_bank'].astype('category').cat.codes
    df['account_id_encoded'] = df['account_number'].astype('category').cat.codes
    
    # Set time index for window calculations
    temp_df = df.set_index('timestamp')
    
    # 3. Recompute the 5 exact behavioral pillars
    user_counts = temp_df.groupby('account_number')['amount'].rolling('1h', closed='left').count().reset_index(level=0, drop=True).fillna(0)
    device_counts = temp_df.groupby('clean_device_id')['amount'].rolling('1h', closed='left').count().reset_index(level=0, drop=True).fillna(0)
    user_7d_avg = temp_df.groupby('account_number')['amount'].rolling('7d', closed='left').mean().reset_index(level=0, drop=True)
    
    bank_diversity = temp_df.groupby('account_number')['bank_id_encoded'].rolling('1h', closed='left').agg(lambda x: len(set(x))).reset_index(level=0, drop=True).fillna(1)
    device_pooling = temp_df.groupby('clean_device_id')['account_id_encoded'].rolling('24h', closed='left').agg(lambda x: len(set(x))).reset_index(level=0, drop=True).fillna(1)
    
    # Map back to main frame
    df['user_tx_count_1h'] = user_counts.values
    df['device_tx_count_1h'] = device_counts.values
    df['unique_dest_banks_1h'] = bank_diversity.values
    df['accounts_per_device_24h'] = device_pooling.values
    df['amount_vs_avg_7d'] = df['amount'] / user_7d_avg.values
    df['amount_vs_avg_7d'] = df['amount_vs_avg_7d'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    
    # Clean up tracking features
    df.drop(columns=['clean_device_id', 'bank_id_encoded', 'account_id_encoded'], inplace=True)
    
    # 4. Extract the chronological test set (Last 20%)
    print("📋 Extracting testing split footprint...")
    test_size = int(len(df) * 0.2)
    test_df = df.tail(test_size)
    
    X_test = test_df.drop(columns=['is_fraud', 'transaction_id', 'customer_name', 'narration'], errors='ignore')
    y_test = test_df['is_fraud'].values
    
    # 5. Load the trained pipeline binary model
    try:
        pipeline = joblib.load(model_path)
        print("🤖 Trained production pipeline loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Error: Could not find trained model at {model_path}. Enforce correct path naming!")
        return
    
    print("🔮 Generating raw prediction probabilities...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # 6. Execute custom threshold array search
    thresholds = np.arange(0.05, 0.95, 0.05)
    results = []
    
    print("\n| Threshold | Fraud Precision | Fraud Recall | Fraud F1-Score |")
    print("|-----------|-----------------|--------------|----------------|")
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])
        
        fraud_prec = precision[1]
        fraud_rec = recall[1]
        fraud_f1 = f1[1]
        
        results.append({
            "threshold": thresh,
            "precision": fraud_prec,
            "recall": fraud_rec,
            "f1_score": fraud_f1
        })
        
        print(f"|   {thresh:.2f}    |      {fraud_prec:.2f}       |     {fraud_rec:.2f}     |      {fraud_f1:.2f}      |")
        
    best_run = max(results, key=lambda x: x['f1_score'])
    print("\n🎯 OPTIMAL SELECTION MATRIX FOUND:")
    print(f" -> Recommended Threshold: {best_run['threshold']:.2f}")
    print(f" -> Expected Max F1-Score: {best_run['f1_score']:.4f}")
    print(f" -> Resulting Precision: {best_run['precision']:.2f} | Recall: {best_run['recall']:.2f}")

if __name__ == "__main__":
    run_threshold_optimization()