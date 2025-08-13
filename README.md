
# telco-churn-mlops — IBM Telco Customer Churn (End-to-End)

An end-to-end ML project using the **IBM Telco Customer Churn** dataset from Kaggle to demonstrate the full pipeline:
data ingestion → EDA → feature engineering → modeling → evaluation → **Streamlit app** deployment.

Dataset: https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

---

## 1) Create repo on GitHub
1. Create an empty repo named **telco-churn-mlops** (no README, no .gitignore).
2. Locally:
```bash
git clone <YOUR_EMPTY_REPO_URL>
cd telco-churn-mlops
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2) Download the Kaggle dataset

### Option A — Kaggle API (recommended)
1. Get an API token from Kaggle (Profile → Account → Create New Token). Save `kaggle.json` to:
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<you>\.kaggle\kaggle.json`
2. Run:
```bash
python scripts/download_kaggle_data.py
```
This downloads and unzips files under `data/raw/`.

### Option B — Manual
Download from the link above and place CSV(s) into `data/raw/`.

---

## 3) Train & run the app
```bash
# Quick start: train on the Kaggle CSV and launch Streamlit
python -m src.models.train --csv data/raw/Telco-Customer-Churn.csv
streamlit run app/Home.py
```

---

## Project layout
```
data/
  raw/        # Kaggle CSVs
  processed/
notebooks/    # EDA & experiments
src/
  data/ingest.py
  features/build_features.py
  models/train.py
  models/predict.py
  models/metrics.py
  utils/io.py
  utils/validation.py
app/
  Home.py
tests/
models/       # saved artifacts (gitignored)
scripts/
  download_kaggle_data.py
```

## Notes
- Primary model metric: **PR-AUC** (class imbalance) + calibration.
- Business metric: **profit curve** based on outreach cost + incentive + saved margin.
- Use SHAP for interpretability (add next step).
