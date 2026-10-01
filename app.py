"""
app.py
------
Streamlit web application for Credit Card Fraud Detection.

Run locally:
    streamlit run app.py
"""

import pathlib

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔐",
    layout="wide",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "⚠️  Model file not found. Please run `python train_model.py` first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


meta     = load_model()
pipeline = meta["pipeline"]
features = meta["features"]
cat_cols = meta["cat_cols"]
num_cols = meta["num_cols"]
bool_cols = meta["bool_cols"]
cat_values = meta["cat_values"]
classes  = meta["classes"]   # ["Legitimate", "Fraud"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔐 Credit Card Fraud Detection")
st.markdown(
    """
    Enter the transaction details below and click **Predict** to find out whether
    the transaction is **Legitimate** or **Fraudulent**.
    
    > Model: **{}** &nbsp;|&nbsp; Task: **Binary Classification** &nbsp;|&nbsp;
    > Target: `is_fraud`
    """.format(meta["model_name"])
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📝 Transaction Details")

col1, col2, col3 = st.columns(3)

input_data = {}

# ─ Numerical inputs ───────────────────────────────────────────────────────────
with col1:
    st.markdown("**Amount & Balance**")
    input_data["amount_usd"] = st.number_input(
        "Transaction Amount (USD)", min_value=0.0, value=50.0, step=1.0
    )
    input_data["account_balance_usd"] = st.number_input(
        "Account Balance (USD)", min_value=0.0, value=2000.0, step=100.0
    )
    input_data["distance_from_home_km"] = st.number_input(
        "Distance from Home (km)", min_value=0.0, value=10.0, step=1.0
    )
    input_data["hours_since_last_txn"] = st.number_input(
        "Hours Since Last Transaction", min_value=0.0, value=5.0, step=0.5
    )
    input_data["txn_count_last_24h"] = st.number_input(
        "Transactions in Last 24h", min_value=0, value=3, step=1
    )

with col2:
    st.markdown("**Card & Customer Info**")
    input_data["card_age_months"] = st.number_input(
        "Card Age (months)", min_value=0, value=24, step=1
    )
    input_data["customer_age"] = st.number_input(
        "Customer Age", min_value=18, max_value=100, value=35, step=1
    )
    input_data["velocity_score"] = st.number_input(
        "Velocity Score", min_value=0.0, value=20.0, step=1.0
    )
    input_data["merchant_risk_score"] = st.number_input(
        "Merchant Risk Score", min_value=0.0, max_value=100.0, value=30.0, step=1.0
    )
    input_data["prior_disputes"] = st.number_input(
        "Prior Disputes", min_value=0, value=0, step=1
    )
    input_data["cvv_retry_count"] = st.number_input(
        "CVV Retry Count", min_value=0, max_value=10, value=0, step=1
    )
    input_data["time_of_day_hour"] = st.slider(
        "Time of Day (Hour 0–23)", min_value=0, max_value=23, value=12
    )
    input_data["day_of_week"] = st.selectbox(
        "Day of Week", options=list(range(7)),
        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
    )

with col3:
    st.markdown("**Categorical Features**")
    for col in cat_cols:
        options = cat_values.get(col, [])
        label = col.replace("_", " ").title()
        input_data[col] = st.selectbox(label, options=options)

    st.markdown("**Boolean / Flag Features**")
    bool_labels = {
        "is_foreign_transaction":    "Foreign Transaction",
        "is_new_merchant":           "New Merchant",
        "used_vpn":                  "Used VPN",
        "ip_country_mismatch":       "IP Country Mismatch",
        "billing_shipping_mismatch": "Billing/Shipping Mismatch",
        "is_ai_generated_scam_attempt": "AI-Generated Scam Attempt",
    }
    for col in bool_cols:
        label = bool_labels.get(col, col.replace("_", " ").title())
        input_data[col] = int(st.checkbox(label, value=False))

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", type="primary", use_container_width=True):
    # Build DataFrame in the exact column order the pipeline expects
    row = pd.DataFrame([{col: input_data[col] for col in features}])

    prediction  = pipeline.predict(row)[0]
    probability = pipeline.predict_proba(row)[0]

    fraud_prob = probability[1] * 100
    legit_prob = probability[0] * 100

    st.divider()
    st.subheader("🎯 Prediction Result")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error(f"### 🚨 FRAUD DETECTED")
            st.metric("Fraud Probability",    f"{fraud_prob:.1f}%")
            st.metric("Legitimate Probability", f"{legit_prob:.1f}%")
        else:
            st.success(f"### ✅ LEGITIMATE TRANSACTION")
            st.metric("Legitimate Probability", f"{legit_prob:.1f}%")
            st.metric("Fraud Probability",      f"{fraud_prob:.1f}%")

    with res_col2:
        st.markdown("**Confidence Breakdown**")
        conf_df = pd.DataFrame(
            {"Class": classes, "Probability (%)": [legit_prob, fraud_prob]}
        )
        st.bar_chart(conf_df.set_index("Class"))

    with st.expander("📋 Input Features Submitted"):
        st.dataframe(row.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Credit Card Fraud Detection · Powered by scikit-learn & Streamlit · "
    f"Model: {meta['model_name']}"
)
