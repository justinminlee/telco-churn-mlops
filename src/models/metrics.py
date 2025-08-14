import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score
)

def pr_auc(y_true, y_prob): return average_precision_score(y_true, y_prob)
def roc_auc(y_true, y_prob): return roc_auc_score(y_true, y_prob)

def precision_recall_f1(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    )

def profit_at_threshold(y_true, y_prob, threshold, margin, incentive, outreach):
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_true==1) & (y_pred==1)).sum()
    fp = ((y_true==0) & (y_pred==1)).sum()
    return tp*margin - (tp+fp)*(incentive+outreach)

def best_profit_threshold(y_true, y_prob, margin, incentive, outreach):
    ps, rs, ts = precision_recall_curve(y_true, y_prob)
    thresholds = ts if ts.size else np.array([0.5])
    profits = [profit_at_threshold(y_true, y_prob, t, margin, incentive, outreach) for t in thresholds]
    i = int(np.argmax(profits))
    return float(thresholds[i]), float(profits[i])
