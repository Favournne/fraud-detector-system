import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

def extract_feature_importance(model_path=r"C:\Users\USER\fraud_project\model_\models\fraud_pipeline_v1.0.pkl"):
    print("Initiating Global Feature Architecture Extraction...")
    print(f"Target Path: {model_path}")
    
    # Verify file existence explicitly
    if not os.path.exists(model_path):
        print(f"Error: The model file does not exist at {model_path}")
        print("Please check your models folder and verify the exact filename!")
        return

    try:
        pipeline = joblib.load(model_path)
        print("Trained production pipeline successfully decrypted!")
    except Exception as e:
        print(f"Error loading pipeline: {str(e)}")
        return

    # Extract components from scikit-learn pipeline
    try:
        transformer = pipeline.steps[0][1]
        xgb_model = pipeline.steps[-1][1]
        
        feature_names = transformer.engineered_feature_names
        importances = xgb_model.feature_importances_
        
        print(f"Found {len(feature_names)} engineered features mapped in the model.")
    except Exception as e:
        print(f"Error extracting structural steps from pipeline: {str(e)}")
        print("Ensure your pipeline matches the [transformer, model] structure.")
        return
    
    # Construct rank matrix dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    # Enforce directory existence for report figures
    os.makedirs("reports/figures", exist_ok=True)
    
    # Plotting horizontal bar hierarchy
    plt.figure(figsize=(11, 7))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='#117A65', edgecolor='black')
    plt.xlabel('Gini Importance / Information Gain Weight')
    plt.ylabel('Engineered Feature Dimension')
    plt.title('Production Fraud Model - Global Feature Architecture (XGBoost)', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    output_img = "reports/figures/feature_importance.png"
    plt.savefig(output_img, dpi=300)
    print(f"Success! Feature importance summary graph saved to: {output_img}")
    
    print("\n================ GLOBAL FEATURE IMPORTANCE RANKS ================")
    # Print in descending order (highest importance first)
    ranked_df = importance_df.sort_values(by='Importance', ascending=False)
    for idx, row in ranked_df.iterrows():
        print(f"{row['Feature'].ljust(30)} : {row['Importance']:.4f}")
    print("=================================================================\n")

if __name__ == "__main__":
    extract_feature_importance()