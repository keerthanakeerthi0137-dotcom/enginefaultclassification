import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Engine Fault Prediction", page_icon="🚗")

@st.cache_resource
def load_artifacts():
    model = joblib.load("engine_fault_pipeline.pkl")
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)
    return model, feature_names

model, feature_names = load_artifacts()

st.title("🚗 Engine Fault Prediction")
st.caption("Your trained pipeline (Scaler → PCA → LDA → Classifier) runs entirely in the cloud.")

tab1, tab2 = st.tabs(["📂 Upload CSV", "🧮 Single-row Input"])

# --------- CSV UPLOAD ----------
with tab1:
    st.write("**Your CSV must include these columns:**")
    st.code(", ".join(feature_names), language="text")

    file = st.file_uploader("Upload CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)

        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            st.error(f"Missing columns in your CSV: {missing}")
        else:
            X = df[feature_names].copy()
            X = X.apply(pd.to_numeric, errors="coerce")
            if X.isna().any().any():
                X = X.fillna(X.mean())

            preds = model.predict(X)

            # Try probabilities if supported
            try:
                proba = model.predict_proba(X)
                top_conf = np.max(proba, axis=1)
                df["Prediction"] = preds
                df["Confidence"] = top_conf
            except Exception:
                df["Prediction"] = preds

            st.success("✅ Predictions complete!")
            st.dataframe(df.head(50), use_container_width=True)
            st.download_button(
                "Download results",
                df.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )

# --------- SINGLE ROW ----------
with tab2:
    st.write("Enter one row of feature values:")
    cols = st.columns(3)
    values = []
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            values.append(st.number_input(feat, value=0.0, step=0.1, format="%.6f"))

    if st.button("🔮 Predict single row"):
        X_row = np.array(values, dtype=float).reshape(1, -1)
        pred = model.predict(X_row)[0]
        msg = f"Prediction: {pred}"
        try:
            conf = float(np.max(model.predict_proba(X_row)))
            msg += f"  (confidence: {conf:.2%})"
        except Exception:
            pass
        st.success(msg)
