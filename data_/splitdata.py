import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def execute_stratified_split(raw_data_path: str, output_dir: str):
    logging.info("Loading raw dataset for splitting...")
    
    # FIXED: Use the parameter instead of hardcoding
    dat = pd.read_csv(raw_data_path)

    df_filtered = dat[dat['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    
    logging.info(f"Filtered dataset shape: {df_filtered.shape}")
    
    # Target and Stratification vector
    y = df_filtered['isFraud']
    
    # 1. First Split: Separate Test Set (15%)
    logging.info("Executing initial split for Holdout Test Set (15%)...")
    df_train_val, df_test = train_test_split(
        df_filtered, 
        test_size=0.15, 
        stratify=y, 
        random_state=42
    )
    
    # 2. Second Split: Separate Train (70% total) and Validation (15% total)
    logging.info("Executing secondary split for Validation Set (15%)...")
    df_train, df_val = train_test_split(
        df_train_val, 
        test_size=0.1765,  # ~15% of original total
        stratify=df_train_val['isFraud'], 
        random_state=42
    )
    
    # Save the splits to disk
    os.makedirs(output_dir, exist_ok=True)
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    logging.info(f"Splits locked successfully! Train: {df_train.shape[0]:,}, Val: {df_val.shape[0]:,}, Test: {df_test.shape[0]:,}")

if __name__ == "__main__":
 
    RAW_PATH = "paysim.csv"
    
    #  FIXED: Save splits inside your data_ folder
    OUTPUT_DIR =  "splits"
    
    execute_stratified_split(RAW_PATH, OUTPUT_DIR)