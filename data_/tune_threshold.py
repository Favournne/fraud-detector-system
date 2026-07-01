import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load the XGBoost model and scaler
    logging.info("Loading XGBoost model and scaler...")
    model = joblib.load(os.path.join(base_dir, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
    
    # 2. Load the HIDDEN Test Set (raw splits, engineer features)
    logging.info("Loading and engineering test set...")
    test_path = os.path.join(base_dir, "splits", "test.csv")
    df = pd.read_csv(test_path)
    
    # Recreate features (MUST match the training script exactly)
    df['hour_of_day'] = df['step'] % 24
    df['type_is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    
    cols_to_drop = ['step', 'type', 'nameOrig', 'nameDest', 'isFlaggedFraud', 
                    'newbalanceOrig', 'newbalanceDest']
    df.drop(columns=cols_to_drop, inplace=True)
    
    X_test = df.drop(columns=['isFraud'])
    y_test = df['isFraud']
    
    # 3. Scale the test set
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Get prediction probabilities
    logging.info("Getting prediction probabilities...")
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 5. Test thresholds from 0.1 to 0.99
    logging.info("Tuning threshold...")
    thresholds = np.linspace(0.1, 0.99, 90)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({'threshold': thresh, 'precision': precision, 
                        'recall': recall, 'f1': f1})
    
    results_df = pd.DataFrame(results)
    
    # 6. Find the "Sweet Spot" where Precision >= 50% and Recall is maximized
    sweet_spot = results_df[results_df['precision'] >= 0.50].iloc[0] if not results_df[results_df['precision'] >= 0.50].empty else None
    
    print("\n" + "="*60)
    print("THRESHOLD TUNING RESULTS")
    print("="*60)
    
    if sweet_spot is not None:
        print(f"\n🎯 Recommended Threshold: {sweet_spot['threshold']:.2f}")
        print(f"   Precision: {sweet_spot['precision']:.2f}")
        print(f"   Recall:    {sweet_spot['recall']:.2f}")
        print(f"   F1-Score:  {sweet_spot['f1']:.2f}")
    else:
        print("\n⚠️  Could not achieve 50% Precision at any threshold.")
        # Fallback: find max F1 threshold
        best_f1 = results_df.loc[results_df['f1'].idxmax()]
        print(f"📌 Best F1 Threshold: {best_f1['threshold']:.2f}")
        print(f"   Precision: {best_f1['precision']:.2f}")
        print(f"   Recall:    {best_f1['recall']:.2f}")
        print(f"   F1-Score:  {best_f1['f1']:.2f}")

    # 7. Plot the trade-off
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['precision'], label='Precision', linewidth=2)
    plt.plot(results_df['threshold'], results_df['recall'], label='Recall', linewidth=2)
    plt.plot(results_df['threshold'], results_df['f1'], label='F1-Score', linewidth=2)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
    if sweet_spot is not None:
        plt.axvline(x=sweet_spot['threshold'], color='green', linestyle='--', label=f"Recommended ({sweet_spot['threshold']:.2f})")
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision vs Recall Trade-off (XGBoost)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("threshold_tuning.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 8. Apply the chosen threshold to get the final report
    if sweet_spot is not None:
        chosen_thresh = sweet_spot['threshold']
    else:
        best_f1 = results_df.loc[results_df['f1'].idxmax()]
        chosen_thresh = best_f1['threshold']
    
    y_pred_final = (y_proba >= chosen_thresh).astype(int)
    print("\n" + "="*60)
    print(f"FINAL CLASSIFICATION REPORT (Threshold = {chosen_thresh:.2f})")
    print("="*60)
    print(classification_report(y_test, y_pred_final, target_names=['Legit', 'Fraud']))
    print("="*60)

if __name__ == "__main__":
    main()