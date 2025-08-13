import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Churn"
ID_COLS = ["customerID"]
NUM_FIXES = ["TotalCharges"]
CONTRACT_MAP = {"month-to-month": 1, "one year": 12, "two year": 24}

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Contract length in months
    if "Contract" in df.columns:
        df["Contract_norm"] = df["Contract"].astype(str).str.lower().str.strip()
        df["contract_length_months"] = df["Contract_norm"].map(CONTRACT_MAP).fillna(0).astype(int)
    # Payment risk flag
    if "PaymentMethod" in df.columns:
        df["is_electronic_check"] = (
            df["PaymentMethod"].astype(str).str.lower().str.contains("electronic check")
        ).astype(int)
    # Support proxy
    if "TechSupport" in df.columns:
        df["has_tech_support"] = (
            df["TechSupport"].astype(str).str.strip().str.lower().eq("yes")
        ).astype(int)
    # Tenure bucket
    if "tenure" in df.columns:
        df["tenure_bucket"] = pd.cut(df["tenure"], bins=[-1,6,12,24,48,72,999], labels=False).astype("Int64")
    # Charges per tenure
    if {"MonthlyCharges","tenure","TotalCharges"}.issubset(df.columns):
        df["charges_per_tenure"] = (df["TotalCharges"] / df["tenure"].replace(0,1)).clip(lower=0)
    return df

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip strings
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    # Drop IDs
    for c in ID_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    # Coerce numerics
    for c in NUM_FIXES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Fill numeric NA
    for c in df.select_dtypes(include=["number"]).columns:
        df[c] = df[c].fillna(df[c].median())
    # Map target
    if TARGET_COL in df.columns and df[TARGET_COL].dtype == "object":
        df[TARGET_COL] = df[TARGET_COL].str.lower().isin(["yes","1","true"]).astype(int)
    # Engineered
    df = add_basic_features(df)
    return df

def split_X_y(df: pd.DataFrame):
    if TARGET_COL in df.columns:
        return df.drop(columns=[TARGET_COL]), df[TARGET_COL].astype(int).values
    return df.copy(), None

def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])
    return pre, cat_cols, num_cols
