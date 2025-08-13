
import streamlit as st
import pandas as pd
import numpy as np

from src.models.predict import predict_proba_df, fit_and_save
from src.models.metrics import best_profit_threshold, pr_auc, roc_auc

st.set_page_config(page_title="Churn Predictor — IBM Telco", layout="wide")
st.title("📉 Customer Churn Predictor — IBM Telco")

st.markdown("""
**How to use**
1. Download the Kaggle dataset to `data/raw/` (see README), or upload here.
2. If your CSV includes a **Churn** column, click **Train model**.
3. Score data to view probabilities, metrics, and a profit-optimised threshold.
""")

uploaded = st.file_uploader("Upload CSV to train/score (include 'Churn' to train)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())

    if "Churn" in df.columns and st.button("Train model"):
        fit_and_save(df)
        st.success("Model trained & saved.")

    try:
        probs = predict_proba_df(df)
        st.subheader("Predictions")
        out = df.copy()
        out["churn_prob"] = probs
        st.dataframe(out.sort_values("churn_prob", ascending=False).head(50), use_container_width=True)

        if "Churn" in df.columns:
            y = df["Churn"].astype(int).values if df["Churn"].dtype != 'object' else (df["Churn"].str.lower().isin(["yes","1","true"]).astype(int).values)
            pr = pr_auc(y, probs)
            ra = roc_auc(y, probs)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PR-AUC", f"{pr:.3f}")
            with col2:
                st.metric("ROC-AUC", f"{ra:.3f}")
            margin = st.number_input("Avg monthly margin per retained customer", min_value=0.0, value=50.0, step=1.0)
            incentive = st.number_input("Incentive cost per outreach", min_value=0.0, value=10.0, step=1.0)
            outreach = st.number_input("Outreach cost per customer", min_value=0.0, value=1.0, step=1.0)
            t, p = best_profit_threshold(y, probs, margin, incentive, outreach)
            st.write(f"**Recommended threshold**: {t:.2f} — Estimated profit at threshold: ${p:,.0f}")
            out["action"] = (out["churn_prob"] >= t).astype(int)

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored CSV", csv, "scored_customers.csv", "text/csv")
    except FileNotFoundError:
        st.warning("No saved model yet. Train on this upload (include 'Churn') or run `python -m src.models.train --csv data/raw/<file>.csv`.")
else:
    st.info("Upload the Kaggle CSV or place it under data/raw/ and use the CLI to train.")
