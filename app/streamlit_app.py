# ---------------------------------------------------------
# UPI Shield AI — Transaction Risk Scoring System
# Streamlit Application
# ---------------------------------------------------------
# This Streamlit app loads the trained fraud detection model
# and predicts whether a digital payment transaction is safe,
# suspicious, or high-risk.
#
# Inputs:
# - Transaction type
# - Transaction amount
# - Sender old/new balance
# - Receiver old/new balance
#
# Outputs:
# - Prediction
# - Fraud probability
# - Risk score
# - Risk level
# - Risk reasons
# - Recommended action
# ---------------------------------------------------------

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
# This must be one of the first Streamlit commands.
# It controls browser title, icon, and layout.
# ---------------------------------------------------------

st.set_page_config(
    page_title="UPI Shield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------
# This makes the app look cleaner and more portfolio-ready.
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 18px;
        color: #4b5563;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: #f8fafc;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
    }

    .metric-label {
        font-size: 15px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #111827;
    }

    .low-risk {
        background-color: #dcfce7;
        color: #166534;
        padding: 12px 16px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }

    .medium-risk {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 12px 16px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }

    .high-risk {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 12px 16px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }

    .info-box {
        background-color: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 15px;
        border-radius: 8px;
        color: #1e3a8a;
    }

    .warning-box {
        background-color: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 15px;
        border-radius: 8px;
        color: #7c2d12;
    }

    .success-box {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 15px;
        border-radius: 8px;
        color: #14532d;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------
# The app file is inside app/.
# So PROJECT_ROOT goes one folder up from app/.
# ---------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
FEATURE_METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "feature_list.json")
MODEL_COMPARISON_PATH = os.path.join(PROJECT_ROOT, "reports", "04_model_training_comparison.csv")
FINAL_METRICS_PATH = os.path.join(PROJECT_ROOT, "reports", "05_final_evaluation_metrics.csv")


# ---------------------------------------------------------
# Load Model and Metadata
# ---------------------------------------------------------
# st.cache_resource prevents repeated loading of model files.
# ---------------------------------------------------------

@st.cache_resource
def load_model_bundle():
    """
    Loads the trained best model bundle from the models folder.
    """
    if not os.path.exists(MODEL_PATH):
        st.error("best_model.pkl not found. Please check the models folder.")
        st.stop()

    model_bundle = joblib.load(MODEL_PATH)
    return model_bundle


@st.cache_resource
def load_feature_metadata():
    """
    Loads feature metadata saved during feature engineering.
    """
    if not os.path.exists(FEATURE_METADATA_PATH):
        st.error("feature_list.json not found. Please check data/processed folder.")
        st.stop()

    with open(FEATURE_METADATA_PATH, "r") as file:
        metadata = json.load(file)

    return metadata


@st.cache_data
def load_model_comparison():
    """
    Loads model comparison report if available.
    """
    if os.path.exists(MODEL_COMPARISON_PATH):
        return pd.read_csv(MODEL_COMPARISON_PATH)
    return None


@st.cache_data
def load_final_metrics():
    """
    Loads final model evaluation metrics if available.
    """
    if os.path.exists(FINAL_METRICS_PATH):
        return pd.read_csv(FINAL_METRICS_PATH)
    return None


model_bundle = load_model_bundle()
feature_metadata = load_feature_metadata()

best_model_name = model_bundle["model_name"]
model = model_bundle["model"]
scaler = model_bundle["scaler"]
requires_scaling = model_bundle["requires_scaling"]
features = model_bundle["features"]

high_amount_threshold = feature_metadata["high_amount_threshold"]
high_risk_types = feature_metadata["high_risk_types"]


# ---------------------------------------------------------
# Helper Function: Risk Level
# ---------------------------------------------------------

def get_risk_level(risk_score):
    """
    Converts risk score into risk level.
    """
    if risk_score <= 30:
        return "Low Risk"
    elif risk_score <= 70:
        return "Medium Risk"
    else:
        return "High Risk"


# ---------------------------------------------------------
# Helper Function: Risk Level CSS Class
# ---------------------------------------------------------

def get_risk_css_class(risk_level):
    """
    Returns CSS class name based on risk level.
    """
    if risk_level == "Low Risk":
        return "low-risk"
    elif risk_level == "Medium Risk":
        return "medium-risk"
    else:
        return "high-risk"


# ---------------------------------------------------------
# Helper Function: Recommended Action
# ---------------------------------------------------------

def get_recommended_action(risk_level):
    """
    Returns recommended action based on risk level.
    """
    if risk_level == "Low Risk":
        return "Allow transaction."
    elif risk_level == "Medium Risk":
        return "Ask for additional verification."
    else:
        return "Hold transaction and perform manual verification."


# ---------------------------------------------------------
# Helper Function: Feature Preparation
# ---------------------------------------------------------
# This function must match Notebook 3 feature engineering.
# ---------------------------------------------------------

def prepare_transaction_features(
    step,
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest
):
    """
    Converts raw transaction input into model-ready features.
    """

    input_df = pd.DataFrame([{
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Balance difference features
    input_df["balanceDiffOrig"] = input_df["oldbalanceOrg"] - input_df["newbalanceOrig"]
    input_df["balanceDiffDest"] = input_df["newbalanceDest"] - input_df["oldbalanceDest"]

    # Balance error features
    input_df["errorBalanceOrig"] = (
        input_df["oldbalanceOrg"] - input_df["amount"] - input_df["newbalanceOrig"]
    )

    input_df["errorBalanceDest"] = (
        input_df["oldbalanceDest"] + input_df["amount"] - input_df["newbalanceDest"]
    )

    # Binary risk indicators
    input_df["isZeroBalanceAfterTransaction"] = np.where(
        input_df["newbalanceOrig"] == 0,
        1,
        0
    )

    input_df["isHighAmount"] = np.where(
        input_df["amount"] >= high_amount_threshold,
        1,
        0
    )

    input_df["isHighRiskType"] = 1 if transaction_type in high_risk_types else 0

    # Time feature
    input_df["hourOfDay"] = input_df["step"] % 24

    # Log features
    input_df["logAmount"] = np.log1p(input_df["amount"])
    input_df["logBalanceDiffOrig"] = np.log1p(np.abs(input_df["balanceDiffOrig"]))
    input_df["logBalanceDiffDest"] = np.log1p(np.abs(input_df["balanceDiffDest"]))
    input_df["logErrorBalanceOrig"] = np.log1p(np.abs(input_df["errorBalanceOrig"]))
    input_df["logErrorBalanceDest"] = np.log1p(np.abs(input_df["errorBalanceDest"]))

    # One-hot transaction type columns
    transaction_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

    for t_type in transaction_types:
        input_df[f"type_{t_type}"] = 1 if transaction_type == t_type else 0

    # Ensure all features exist
    for feature in features:
        if feature not in input_df.columns:
            input_df[feature] = 0

    # Maintain same feature order as training
    input_df = input_df[features]

    return input_df


# ---------------------------------------------------------
# Helper Function: Risk Reasons
# ---------------------------------------------------------

def generate_risk_reasons(
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    risk_score
):
    """
    Generates explainable risk reasons.
    """

    reasons = []

    balance_diff_orig = oldbalanceOrg - newbalanceOrig
    error_orig = oldbalanceOrg - amount - newbalanceOrig
    error_dest = oldbalanceDest + amount - newbalanceDest

    if transaction_type in high_risk_types:
        reasons.append(f"{transaction_type} is considered a higher-risk transaction type.")

    if amount >= high_amount_threshold:
        reasons.append("Transaction amount is higher than the high-amount threshold learned from the dataset.")

    if newbalanceOrig == 0 and oldbalanceOrg > 0:
        reasons.append("Sender balance became zero after the transaction.")

    if abs(error_orig) > 1:
        reasons.append("Sender balance movement does not perfectly match the transaction amount.")

    if abs(error_dest) > 1:
        reasons.append("Receiver balance movement does not perfectly match the transaction amount.")

    if balance_diff_orig >= amount and amount > 0:
        reasons.append("Sender balance reduction is equal to or greater than the transaction amount.")

    if risk_score > 70:
        reasons.append("Model predicted high fraud probability for this transaction.")
    elif risk_score > 30:
        reasons.append("Model predicted medium fraud probability for this transaction.")
    else:
        reasons.append("Model predicted low fraud probability for this transaction.")

    if len(reasons) == 0:
        reasons.append("No major suspicious rule-based pattern detected.")

    return reasons


# ---------------------------------------------------------
# Helper Function: Prediction
# ---------------------------------------------------------

def predict_transaction_risk(
    step,
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest
):
    """
    Predicts transaction risk using the trained model.
    """

    input_features = prepare_transaction_features(
        step=step,
        transaction_type=transaction_type,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest
    )

    if requires_scaling:
        input_model = scaler.transform(input_features)
    else:
        input_model = input_features

    prediction = model.predict(input_model)[0]

    if hasattr(model, "predict_proba"):
        fraud_probability = model.predict_proba(input_model)[0][1]
    else:
        fraud_probability = float(prediction)

    risk_score = round(float(fraud_probability) * 100, 2)
    risk_level = get_risk_level(risk_score)
    recommended_action = get_recommended_action(risk_level)

    reasons = generate_risk_reasons(
        transaction_type=transaction_type,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest,
        risk_score=risk_score
    )

    result = {
        "prediction": "Suspicious / Fraud" if prediction == 1 else "Normal / Safe",
        "fraud_probability": round(float(fraud_probability), 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "reasons": reasons,
        "input_features": input_features
    }

    return result


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.title("🛡️ UPI Shield AI")

    st.markdown(
        """
        **Transaction Risk Scoring System**

        This app predicts whether a digital payment transaction is safe or suspicious using a trained machine learning model.
        """
    )

    st.divider()

    st.subheader("Model Info")
    st.write("Best Model:", best_model_name)
    st.write("Total Features:", len(features))
    st.write("Scaling Required:", "Yes" if requires_scaling else "No")

    st.divider()

    st.subheader("Risk Score Guide")
    st.markdown(
        """
        - **0–30:** Low Risk  
        - **31–70:** Medium Risk  
        - **71–100:** High Risk
        """
    )

    st.divider()

    st.caption("Built for AIML portfolio project")


# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------

st.markdown('<div class="main-title">🛡️ UPI Shield AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Machine Learning based Transaction Fraud Risk Scoring System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    Enter transaction details below. The system will predict fraud probability, risk score, risk level,
    reasons, and recommended action.
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------

st.subheader("🔍 Transaction Risk Check")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
            index=1
        )

        step = st.number_input(
            "Transaction Step / Hour",
            min_value=1,
            max_value=1000,
            value=25,
            step=1
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=95000.0,
            step=1000.0,
            format="%.2f"
        )

    with col2:
        oldbalanceOrg = st.number_input(
            "Sender Old Balance",
            min_value=0.0,
            value=95000.0,
            step=1000.0,
            format="%.2f"
        )

        newbalanceOrig = st.number_input(
            "Sender New Balance",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            format="%.2f"
        )

        oldbalanceDest = st.number_input(
            "Receiver Old Balance",
            min_value=0.0,
            value=2000.0,
            step=1000.0,
            format="%.2f"
        )

        newbalanceDest = st.number_input(
            "Receiver New Balance",
            min_value=0.0,
            value=97000.0,
            step=1000.0,
            format="%.2f"
        )

    submitted = st.form_submit_button("Check Transaction Risk")


# ---------------------------------------------------------
# Prediction Output
# ---------------------------------------------------------

if submitted:
    result = predict_transaction_risk(
        step=step,
        transaction_type=transaction_type,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest
    )

    st.write("")
    st.subheader("📊 Prediction Result")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Prediction</div>
                <div class="metric-value">{result["prediction"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Fraud Probability</div>
                <div class="metric-value">{result["fraud_probability"] * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{result["risk_score"]}/100</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_col4:
        risk_class = get_risk_css_class(result["risk_level"])
        st.markdown(
            f"""
            <div class="{risk_class}">
                {result["risk_level"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    if result["risk_level"] == "High Risk":
        st.markdown(
            f"""
            <div class="warning-box">
            <b>Recommended Action:</b> {result["recommended_action"]}
            </div>
            """,
            unsafe_allow_html=True
        )
    elif result["risk_level"] == "Medium Risk":
        st.markdown(
            f"""
            <div class="warning-box">
            <b>Recommended Action:</b> {result["recommended_action"]}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="success-box">
            <b>Recommended Action:</b> {result["recommended_action"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.subheader("🧠 Why This Risk Level?")

    for reason in result["reasons"]:
        st.write(f"- {reason}")

    with st.expander("View Model Input Features"):
        st.dataframe(result["input_features"])


# ---------------------------------------------------------
# Example Transactions
# ---------------------------------------------------------

st.divider()
st.subheader("🧪 Example Transactions to Try")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    st.markdown(
        """
        **Low-Risk Example**
        - Type: PAYMENT
        - Amount: 2,500
        - Sender Old: 50,000
        - Sender New: 47,500
        - Receiver Old: 10,000
        - Receiver New: 12,500
        """
    )

with example_col2:
    st.markdown(
        """
        **Medium-Risk Example**
        - Type: CASH_OUT
        - Amount: 35,000
        - Sender Old: 80,000
        - Sender New: 45,000
        - Receiver Old: 5,000
        - Receiver New: 40,000
        """
    )

with example_col3:
    st.markdown(
        """
        **High-Risk Example**
        - Type: TRANSFER
        - Amount: 95,000
        - Sender Old: 95,000
        - Sender New: 0
        - Receiver Old: 2,000
        - Receiver New: 97,000
        """
    )


# ---------------------------------------------------------
# Model Performance Section
# ---------------------------------------------------------

st.divider()
st.subheader("📈 Model Performance")

final_metrics_df = load_final_metrics()
model_comparison_df = load_model_comparison()

perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.markdown("### Final Best Model Metrics")
    if final_metrics_df is not None:
        st.dataframe(final_metrics_df)
    else:
        st.info("Final evaluation metrics file not found.")

with perf_col2:
    st.markdown("### Model Comparison")
    if model_comparison_df is not None:
        st.dataframe(model_comparison_df)
    else:
        st.info("Model comparison report not found.")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#6b7280; padding:15px;">
    UPI Shield AI — AIML Portfolio Project | Machine Learning + Risk Scoring + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)