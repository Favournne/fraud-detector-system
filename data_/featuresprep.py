import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FeatureEngineeringPipeline:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def transform_split(self, file_name: str) -> pd.DataFrame:
        file_path = os.path.join(self.input_dir, file_name)
        logging.info(f"Loading and dropping noise/leakage columns for: {file_name}")
        
        # Load the split data
        df = pd.read_csv(file_path)
        
        # Drop verified noise, leaks, and high-cardinality text columns
        # NO new features are being created here - just cleaning
        cols_to_drop = [
            'step', 'type', 'nameOrig', 'nameDest', 'isFlaggedFraud',
            'newbalanceOrig', 'newbalanceDest'
        ]
        df_transformed = df.drop(columns=cols_to_drop)
        
        logging.info(f"Remaining columns: {df_transformed.columns.tolist()}")
        
        return df_transformed

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        splits = ['train.csv', 'validation.csv', 'test.csv']
        
        for split in splits:
            input_path = os.path.join(self.input_dir, split)
            if not os.path.exists(input_path):
                logging.error(f"Could not find split file: {input_path}")
                continue
                
            processed_df = self.transform_split(split)
            
            output_path = os.path.join(self.output_dir, f"feature_{split}")
            processed_df.to_csv(output_path, index=False)
            logging.info(f"Saved: {output_path} | Final Dimensions: {processed_df.shape}")

if __name__ == "__main__":
    # Dynamic path resolution so it works whether you run from inside data_ or from the parent folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    INPUT_SPLITS = os.path.join(BASE_DIR, "splits")
    OUTPUT_FEATURES = os.path.join(BASE_DIR, "processed_features")
    
    logging.info(f"Looking for splits in: {INPUT_SPLITS}")
    logging.info(f"Saving cleaned data to: {OUTPUT_FEATURES}")
    
    pipeline = FeatureEngineeringPipeline(input_dir=INPUT_SPLITS, output_dir=OUTPUT_FEATURES)
    pipeline.run()