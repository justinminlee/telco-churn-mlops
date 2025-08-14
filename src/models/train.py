import argparse, os, numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..features.build_features import clean_columns, split_X_y, build_preprocessor
from ..models.metrics import pr_auc, roc_auc

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pd_models'))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pd_models', 'model.pkl'))

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        for enc in ("utf-8-sig","latin1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise

def build_pipelines(X):
    pre, _, _ = build_preprocessor(X)
    lr = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=500, class_weight='balanced'))])
    rf = Pipeline([('pre', pre), ('clf', RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42, class_weight='balanced'))])
    xgb = Pipeline([('pre', pre), ('clf', XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, eval_metric='logloss'))])
    return {'logreg': lr, 'random_forest': rf, 'xgboost': xgb}

def cv_scores(model, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:,1]
    return pr_auc(y, probs), roc_auc(y, probs), probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help="Path to CSV/XLSX")
    args = parser.parse_args()

    if not os.path.exists(args.csv): raise FileNotFoundError(args.csv)
    df = read_any(args.csv)
    df = clean_columns(df)
    X, y = split_X_y(df)
    if y is None:
        raise ValueError("Could not find a 'Churn' target column (or variant). Please check your file.")

    models = build_pipelines(X)
    print('Evaluating models (5-fold CV)...')
    results = {}
    for name, pipe in models.items():
        pra, ra, probs = cv_scores(pipe, X, y)
        results[name] = {'pr_auc': pra, 'roc_auc': ra}
        print(f"{name:13s}  PR-AUC={pra:.4f}  ROC-AUC={ra:.4f}")

    best_name = max(results, key=lambda k: results[k]['pr_auc'])
    best_model = models[best_name]
    print(f"\nBest by PR-AUC: {best_name} => {results[best_name]}")

    best_model.fit(X, y)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")

if __name__ == '__main__':
    main()
