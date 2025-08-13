
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "yeanzc/telco-customer-churn-ibm-dataset"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {DATASET} to {OUT_DIR} ...")
    api.dataset_download_files(DATASET, path=OUT_DIR, quiet=False)
    # Find the downloaded zip
    zips = [f for f in os.listdir(OUT_DIR) if f.endswith('.zip')]
    for z in zips:
        zp = os.path.join(OUT_DIR, z)
        print(f"Unzipping {zp} ...")
        with zipfile.ZipFile(zp, 'r') as zf:
            zf.extractall(OUT_DIR)
        os.remove(zp)
    print("Done. Check data/raw/.")

if __name__ == "__main__":
    main()
