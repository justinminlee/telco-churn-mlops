import re
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical names
TARGET_COL = "Churn"
ID_COLS = ["customerID"]
NUM_FIXES = ["TotalCharges"]

# Map common IBM Telco header variants to canonical names
_CANON_MAP = {
    r"^customer\s*id$": "customerID",
    r"^senior\s*citizen$": "SeniorCitizen",
    r"^tenure(\s*months?)?$": "tenure",
    r"^phone\s*service$": "PhoneService",
    r"^multiple\s*lines$": "MultipleLines",
    r"^internet\s*service$": "InternetService",
    r"^online\s*security$": "OnlineSecurity",
    r"^online\s*backup$": "OnlineBackup",
    r"^device\s*protection$": "DeviceProtection",
    r"^tech\s*support$": "TechSupport",
    r"^streaming\s*tv$": "StreamingTV",
    r"^streaming\s*movies$": "StreamingMovies",
    r"^paperless\s*billing$": "PaperlessBilling",
    r"^payment\s*method$": "PaymentMethod",
    r"^monthly\s*charges?$": "MonthlyCharges",
    r"^total\s*charges?$": "TotalCharges",
    r"^contract$": "Contract",
    r"^gender$": "gender",
    r"^partner$": "Partner",
    r"^dependents?$": "Dependents",
    r"^churn$": "Churn",
    r"^churn\s*label$": "Churn",
    r"^churn\s*value$": "Churn",
    r"^customer\s*status$": "CustomerStatus",
    r"^churn\s*score$": "ChurnScore",
    r"^churn\s*category$": "ChurnCategory",
    r"^churn\s*reason$": "ChurnReason",
}

# Contract to months (feature)
_CONTRACT_MAP = {"month-to-month": 1, "one year": 12, "two year": 24}

# Always drop
DROP_ALWAYS = {"Count"}
# Non-actionable / dataset-specific extras to drop
DROP_EXTRA = {"Country","State","City","Zip Code","Lat Long","Latitude","Longitude","CLTV"}
# Other churn-derived (leaky) names
LEAKY_NAMES_LC = {"customerstatus","churnscore","churncategory","churnreason"}
LEAKY_PREFIXES_LC = ("churn ", "churn_")  # e.g., "Churn Label", "Churn Value"

def _strip_lower(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace, map header variants, and ensure unique names."""
    df = df.copy()
    mapped = []
    for c in df.columns:
        cl = _strip_lower(c)
        new = None
        for pat, target in _CANON_MAP.items():
            if re.fullmatch(pat, cl):
                new = target
                break
        mapped.append(new if new else str(c).strip())
    # ensure uniqueness
    counts, unique = {}, []
    for name in mapped:
        if name in counts:
            counts[name] += 1
            unique.append(f"{name}__dup{counts[name]}")
        else:
            counts[name] = 0
            unique.append(name)
    df.columns = unique
    return df

def ensure_target_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COL in df.columns:
        s = df[TARGET_COL]
        if s.dtype == "object":
            df[TARGET_COL] = s.astype(str).str.strip().str.lower().isin(["yes","1","true"]).astype(int)
        else:
            df[TARGET_COL] = (pd.to_numeric(s, errors="coerce").fillna(0) > 0.5).astype(int)
        return df
    lc_map = {str(c).strip().lower(): c for c in df.columns}
    for key in ("churn value", "churn_value", "churn label", "churn_label", "churn"):
        if key in lc_map:
            src = lc_map[key]; s = df[src]
            if s.dtype == "object":
                df[TARGET_COL] = s.astype(str).str.strip().str.lower().isin(["yes","1","true"]).astype(int)
            else:
                df[TARGET_COL] = (pd.to_numeric(s, errors="coerce").fillna(0) > 0.5).astype(int)
            return df
    return df

def drop_leaky_and_useless(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = set()
    for c in df.columns:
        if c in DROP_ALWAYS: to_drop.add(c); continue
        if c != TARGET_COL and c.lower().startswith("churn__dup"): to_drop.add(c); continue
        lc = c.strip().lower()
        if lc in LEAKY_NAMES_LC: to_drop.add(c); continue
        if c != TARGET_COL and lc.startswith(LEAKY_PREFIXES_LC): to_drop.add(c); continue
        if c in DROP_EXTRA: to_drop.add(c); continue
    if to_drop:
        df = df.drop(columns=list(to_drop), errors="ignore")
    return df

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Contract" in df.columns:
        df["Contract_norm"] = df["Contract"].astype(str).str.lower().str.strip()
        df["contract_length_months"] = df["Contract_norm"].map(_CONTRACT_MAP).fillna(0).astype(int)
    if "PaymentMethod" in df.columns:
        df["is_electronic_check"] = (
            df["PaymentMethod"].astype(str).str.lower().str.contains("electronic check")
        ).astype(int)
    if "TechSupport" in df.columns:
        df["has_tech_support"] = df["TechSupport"].astype(str).str.strip().str.lower().eq("yes").astype(int)
    if "tenure" in df.columns:
        df["tenure_bucket"] = pd.cut(df["tenure"], bins=[-1,6,12,24,48,72,999], labels=False).astype("Int64")
    if {"MonthlyCharges","tenure","TotalCharges"}.issubset(df.columns):
        df["charges_per_tenure"] = (df["TotalCharges"] / df["tenure"].replace(0,1)).clip(lower=0)
    return df

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)

    # Strip strings
    obj_cols = [c for c, dt in df.dtypes.items() if dt == "object"]
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # 🔧 Fill missing categoricals BEFORE OHE to avoid isnan/type errors
    if obj_cols:
        df[obj_cols] = df[obj_cols].replace({"": None}).fillna("Unknown")

    # Drop IDs
    for c in ID_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Coerce known numerics
    for c in NUM_FIXES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill numeric NA
    for c, dt in df.dtypes.items():
        if str(dt).startswith(("float", "int")):
            df[c] = df[c].fillna(df[c].median())

    # Target + drops + engineered
    df = ensure_target_column(df)
    df = drop_leaky_and_useless(df)
    df = add_basic_features(df)
    return df

def split_X_y(df: pd.DataFrame):
    df = normalize_column_names(df)
    if TARGET_COL in df.columns:
        return df.drop(columns=[TARGET_COL]), df[TARGET_COL].astype(int).values
    return df.copy(), None

def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    return pre, cat_cols, num_cols
