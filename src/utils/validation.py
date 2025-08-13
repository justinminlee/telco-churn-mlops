from sklearn.model_selection import StratifiedKFold

def get_cv(n_splits=5, seed=42):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
