import streamlit as st, pandas as pd, numpy as np, shap, matplotlib.pyplot as plt
from src.models.predict import predict_proba_df
from src.models.metrics import pr_auc, roc_auc, best_profit_threshold, precision_recall_f1
import joblib

st.set_page_config(page_title="Churn Predictor — IBM Telco", layout="wide")
st.title("📉 Customer Churn Predictor — IBM Telco")

def read_uploaded(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file, engine="openpyxl")
    return pd.read_csv(file)

tab1, tab2, tab3 = st.tabs(["Batch scoring", "Single-customer calculator", "Model explainability (SHAP)"])

with tab1:
    st.subheader("Batch scoring & evaluation")
    uploaded = st.file_uploader("Upload CSV/XLSX to score/evaluate", type=["csv","xlsx","xls"], key="batch")
    if uploaded is not None:
        df = read_uploaded(uploaded)
        st.write("Preview:", df.head())
        probs = predict_proba_df(df)
        out = df.copy()
        out["churn_prob"] = probs
        st.dataframe(out.sort_values("churn_prob", ascending=False).head(50), use_container_width=True)

        if "Churn" in df.columns:
            y = (df["Churn"].astype(str).str.lower().isin(["yes","1","true"]).astype(int).values
                 if df["Churn"].dtype == 'object' else df["Churn"].astype(int).values)
            c1, c2 = st.columns(2)
            with c1: st.metric("PR-AUC", f"{pr_auc(y, probs):.3f}")
            with c2: st.metric("ROC-AUC", f"{roc_auc(y, probs):.3f}")
            st.markdown("**Cost–benefit analysis**")
            margin = st.number_input("Avg monthly margin per retained customer", min_value=0.0, value=50.0, step=1.0)
            incentive = st.number_input("Incentive cost per outreach", min_value=0.0, value=10.0, step=1.0)
            outreach = st.number_input("Outreach cost per customer", min_value=0.0, value=1.0, step=1.0)
            t, p = best_profit_threshold(y, probs, margin, incentive, outreach)
            st.write(f"**Recommended threshold**: {t:.2f} — Estimated profit at threshold: ${p:,.0f}")
            prec, rec, f1 = precision_recall_f1(y, probs, threshold=t)
            st.write(f"Precision={prec:.2f} · Recall={rec:.2f} · F1={f1:.2f} at threshold {t:.2f}")
            out["action"] = (out["churn_prob"] >= t).astype(int)
        st.download_button("Download scored CSV", out.to_csv(index=False).encode("utf-8"),
                           "scored_customers.csv", "text/csv")
    else:
        st.info("Upload the Kaggle CSV/XLSX you trained on to view metrics and scores.")

with tab2:
    st.subheader("Interactive churn probability calculator")
    st.caption("Provide customer details to estimate churn probability with the saved model.")
    colA, colB, colC = st.columns(3)
    with colA:
        gender = st.selectbox("gender", ["Female","Male"])
        SeniorCitizen = st.selectbox("SeniorCitizen", [0,1])
        Partner = st.selectbox("Partner", ["Yes","No"])
        Dependents = st.selectbox("Dependents", ["Yes","No"])
        tenure = st.number_input("tenure", min_value=0, max_value=120, value=5, step=1)
        PhoneService = st.selectbox("PhoneService", ["Yes","No"])
        MultipleLines = st.selectbox("MultipleLines", ["No phone service","No","Yes"])
    with colB:
        InternetService = st.selectbox("InternetService", ["DSL","Fiber optic","No"])
        OnlineSecurity = st.selectbox("OnlineSecurity", ["No internet service","No","Yes"])
        OnlineBackup = st.selectbox("OnlineBackup", ["No internet service","No","Yes"])
        DeviceProtection = st.selectbox("DeviceProtection", ["No internet service","No","Yes"])
        TechSupport = st.selectbox("TechSupport", ["No internet service","No","Yes"])
        StreamingTV = st.selectbox("StreamingTV", ["No internet service","No","Yes"])
        StreamingMovies = st.selectbox("StreamingMovies", ["No internet service","No","Yes"])
    with colC:
        Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"], index=0)
        PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes","No"], index=0)
        PaymentMethod = st.selectbox("PaymentMethod", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, value=70.0, step=1.0)
        TotalCharges = st.number_input("TotalCharges", min_value=0.0, value=500.0, step=10.0)
    if st.button("Calculate churn probability"):
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents,
            "tenure": tenure, "PhoneService": PhoneService, "MultipleLines": MultipleLines,
            "InternetService": InternetService, "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection, "TechSupport": TechSupport, "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies, "Contract": Contract, "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges
        }])
        prob = float(predict_proba_df(row)[0])
        st.success(f"Estimated churn probability: {prob:.3f}")

with tab3:
    st.subheader("Global explainability (SHAP)")
    st.caption("Upload a CSV/XLSX (sampled to 500 rows) and see a global beeswarm plot.")

    shap_file = st.file_uploader("Upload a CSV/XLSX (optional)", type=["csv","xlsx","xls"], key="shap")
    if shap_file is not None:
        data = read_uploaded(shap_file)

        # Use the same cleaning/feature-engineering as training
        from src.features.build_features import clean_columns
        data2 = clean_columns(data)

        import joblib, numpy as np, pandas as pd, shap, matplotlib.pyplot as plt
        model = joblib.load("./pd_models/model.pkl")

        if hasattr(model, 'named_steps') and 'pre' in model.named_steps and 'clf' in model.named_steps:
            pre = model.named_steps['pre']
            clf = model.named_steps['clf']

            # Ensure expected columns exist (Unknown/0) in case upload is missing some
            try:
                cat_cols = list(pre.transformers_[0][2]) if pre.transformers_ and len(pre.transformers_) > 0 else []
                num_cols = list(pre.transformers_[1][2]) if pre.transformers_ and len(pre.transformers_) > 1 else []
            except Exception:
                cat_cols, num_cols = [], []

            def ensure_expected_columns(df, cat_cols, num_cols):
                df = df.copy()
                for c in cat_cols:
                    if c not in df.columns: df[c] = "Unknown"
                for c in num_cols:
                    if c not in df.columns: df[c] = 0
                if cat_cols:
                    df[cat_cols] = df[cat_cols].replace({"": None}).fillna("Unknown")
                    for c in cat_cols: df[c] = df[c].astype(str)
                return df

            data2 = ensure_expected_columns(data2, cat_cols, num_cols)
            X = data2.drop(columns=['Churn'], errors='ignore')

            try:
                # Best: ask sklearn for all output names directly
                feat_names = pre.get_feature_names_out().tolist()
            except Exception:
                # Fallback: cat one-hot names + numeric columns
                try:
                    ohe = pre.named_transformers_['cat']
                    cat_names = ohe.get_feature_names_out(cat_cols).tolist() if cat_cols else []
                except Exception:
                    cat_names = []
                feat_names = cat_names + num_cols

            # Transform, then wrap into a DataFrame with column names
            Xt = pre.transform(X)
            try:
                Xt_df = pd.DataFrame(Xt, columns=feat_names)
            except Exception:
                # If lengths mismatch, fall back to generic names (shouldn’t happen, but safe)
                Xt_df = pd.DataFrame(Xt, columns=[f"f{i}" for i in range(Xt.shape[1])])

            # Sample (optional) for speed
            if Xt_df.shape[0] > 500:
                idx = np.random.RandomState(42).choice(Xt_df.shape[0], 500, replace=False)
                Xt_df = Xt_df.iloc[idx]

            # Explain
            if "xgboost" in str(type(clf)).lower():
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer(Xt_df)
            else:
                explainer = shap.Explainer(clf, Xt_df)
                shap_values = explainer(Xt_df)

            fig = plt.figure()
            shap.plots.beeswarm(shap_values, max_display=15, show=False)
            st.pyplot(fig)
        else:
            st.warning("Saved model is not a pipeline with 'pre' and 'clf'. Retrain with training script.")
    else:
        st.info("Upload a CSV/XLSX to generate SHAP global importance.")