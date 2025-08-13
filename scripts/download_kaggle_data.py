import os
import re
import shutil
import unicodedata
import csv
import numpy as np
import pandas as pd
import kagglehub

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def _sanitize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")

def _excel_to_csv(xlsx_path: str, out_dir: str) -> str:
    """Convert Excel to CSV with minimal transformation:
    - dtype=object to avoid coercion
    - replace NaN with empty string to preserve blanks
    - choose sheet with 'Churn' column if available
    """
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")

    chosen_df = None
    chosen_sheet = None
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl", dtype=object)
        df = df.replace({np.nan: ""})
        if "Churn" in df.columns:
            chosen_df = df
            chosen_sheet = sheet
            break

    if chosen_df is None:
        # fallback to first sheet
        chosen_sheet = xls.sheet_names[0]
        chosen_df = pd.read_excel(xlsx_path, sheet_name=chosen_sheet, engine="openpyxl", dtype=object)
        chosen_df = chosen_df.replace({np.nan: ""})

    # Prefer consistent name; if exists, use a safe fallback
    preferred = os.path.join(out_dir, "Telco-Customer-Churn.csv")
    if not os.path.exists(preferred):
        out_path = preferred
    else:
        base = os.path.splitext(os.path.basename(xlsx_path))[0]
        safe = _sanitize(f"{base}_{chosen_sheet}.csv")
        out_path = os.path.join(out_dir, safe)

    chosen_df.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return out_path

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Download latest version to Kaggle cache
    path = kagglehub.dataset_download("yeanzc/telco-customer-churn-ibm-dataset")
    print("Downloaded to cache:", path)

    converted = []
    copied = []
    for file_name in os.listdir(path):
        src = os.path.join(path, file_name)
        if not os.path.isfile(src):
            continue

        dst = os.path.join(RAW_DIR, file_name)
        shutil.copy2(src, dst)
        copied.append(dst)

        # Auto-convert Excel files to CSV
        lower = file_name.lower()
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            csv_path = _excel_to_csv(dst, RAW_DIR)
            converted.append(csv_path)

    print("Copied files to:", os.path.abspath(RAW_DIR))
    if converted:
        print("Converted the following Excel files to CSV:")
        for p in converted:
            print("  -", os.path.basename(p))

if __name__ == "__main__":
    main()
