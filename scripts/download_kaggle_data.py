import os
import kagglehub
import shutil

# Download latest version
path = kagglehub.dataset_download("yeanzc/telco-customer-churn-ibm-dataset")

# Move CSV files into data/raw
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

for file_name in os.listdir(path):
    src = os.path.join(path, file_name)
    dst = os.path.join(RAW_DIR, file_name)
    if os.path.isfile(src):
        shutil.copy2(src, dst)

print("Dataset files copied to:", os.path.abspath(RAW_DIR))
