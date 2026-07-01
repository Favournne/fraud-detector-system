import kagglehub
import shutil
import os

def sync_paysim_data():
    cache_path = kagglehub.dataset_download("moonknightmarvel/paysim")

    csv_file_path = None
    for filename in os.listdir(cache_path):
        if filename.endswith('.csv'):
            csv_file_path =filename
            break
    if not csv_file_path:
        print("No CSV file found in the dataset!")
        return None 
    target_dir = r"C:\Users\USER\fraud_project\data_"  # Your exact folder
    os.makedirs(target_dir, exist_ok=True) 

    src_file = os.path.join(cache_path, csv_file_path)
    dest_file = os.path.join(target_dir, csv_file_path)

    if os.path.exists(dest_file):
        print(f"File already exists at: {dest_file}")
        print("Skipping copy to preserve existing file.")

    else:
        shutil.copy2(src_file, dest_file)
        print(f"Copied: {csv_file_path} to {target_dir}")

    print(f"CSV file path: {dest_file}")
    return dest_file

if __name__ == "__main__":
    csv_path = sync_paysim_data()