
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Churn"

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize common quirks in the IBM dataset (e.g., 'Yes', 'No', spaces)
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].astype(str).str.strip()
    # Convert target to 0/1 if present
    if TARGET_COL in df.columns and df[TARGET_COL].dtype == 'object':
        df[TARGET_COL] = (df[TARGET_COL].str.lower().isin(["yes", "1", "true"])).astype(int)
    return df

def split_X_y(df: pd.DataFrame):
    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].astype(int).values
        X = df.drop(columns=[TARGET_COL])
    else:
        X = df.copy()
    return X, y

def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])
    return pre, cat_cols, num_cols
