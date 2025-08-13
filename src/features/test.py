import pandas as pd
from build_features import clean_columns
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
df = clean_columns(df)
print("Has Churn?", "Churn" in df.columns, "| dtype:", df["Churn"].dtype if "Churn" in df.columns else None)
print("Sample engineered cols present:",
      [c for c in df.columns if c in ("contract_length_months","is_electronic_check","has_tech_support","tenure_bucket","charges_per_tenure")])
