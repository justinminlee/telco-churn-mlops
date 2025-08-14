import os, joblib, numpy as np, pandas as pd
from ..features.build_features import clean_columns

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pd_models'))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pd_models', 'model.pkl'))

def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No saved model found at models/model.pkl. Train first.")
    return joblib.load(MODEL_PATH)

def _expected_feature_lists(model):
    pre = model.named_steps.get('pre', None)
    if pre is None or not hasattr(pre, "transformers_"):
        raise RuntimeError("Saved model is missing the 'pre' ColumnTransformer. Retrain.")
    cat_cols = list(pre.transformers_[0][2]) if pre.transformers_ and len(pre.transformers_) > 0 else []
    num_cols = list(pre.transformers_[1][2]) if pre.transformers_ and len(pre.transformers_) > 1 else []
    return cat_cols, num_cols

def _ensure_expected_columns(df: pd.DataFrame, cat_cols, num_cols) -> pd.DataFrame:
    df = df.copy()
    # Ensure categoricals exist & have no missing
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "Unknown"
    if cat_cols:
        df[cat_cols] = df[cat_cols].replace({"": None}).fillna("Unknown")
        # Make sure all cats are strings to avoid mixed types
        for c in cat_cols:
            df[c] = df[c].astype(str)
    # Ensure numerics exist
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
    return df

def _sanitize_encoder_categories(pre):

    try:
        ohe = pre.named_transformers_['cat']
    except Exception:
        return  # no categorical transformer

    if not hasattr(ohe, "categories_"):
        return

    new_cats = []
    changed = False
    for arr in ohe.categories_:
        # arr is a numpy array of categories; swap NaN/None with 'Unknown'
        replaced = np.array(
            ["Unknown" if (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else v
             for v in arr],
            dtype=object
        )
        if not np.array_equal(arr, replaced):
            changed = True
        new_cats.append(replaced)

    if changed:
        ohe.categories_ = new_cats

def predict_proba_df(df: pd.DataFrame) -> np.ndarray:
    df = clean_columns(df)  # normalize headers, engineer features, fill some NAs
    model = _load_model()

    # Grab expected raw feature lists from the fitted preprocessor
    pre = model.named_steps['pre']
    cat_cols, num_cols = _expected_feature_lists(model)

    # Ensure inputs match what the model expects
    df = _ensure_expected_columns(df, cat_cols, num_cols)

    # 🔒 Most important bit: sanitize fitted OHE categories to remove NaN
    _sanitize_encoder_categories(pre)

    X = df.drop(columns=['Churn'], errors='ignore')
    return model.predict_proba(X)[:, 1]
