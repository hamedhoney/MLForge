import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from mlforge.deployment import download_model

MODEL_DIR = Path(os.getenv("MLFORGE_MODEL_DIR", "artifacts/tuning"))

st.set_page_config(
    page_title="MLForge | Customer Churn",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the trusted trained model and matching metadata."""
    model_path = MODEL_DIR / "best_model.joblib"
    metadata_path = MODEL_DIR / "metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        download_model()

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())

    return model, metadata


st.title("MLForge")
st.caption("Production MLOps | Customer Churn Prediction")

try:
    model, metadata = load_model()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.header("Model Information")
st.sidebar.write(f"Model: {metadata['model']}")
st.sidebar.write(f"Threshold: {metadata['threshold']:.2f}")
st.sidebar.write(f"Features: {len(metadata['features'])}")

st.subheader("Customer Profile")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, value=12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
        )

    with col2:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=79.5,
        )
        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=950.0,
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True,
    )

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    frame = pd.DataFrame([customer])
    frame = frame.reindex(columns=metadata["features"])

    probability = float(model.predict_proba(frame)[0, 1])
    threshold = float(metadata["threshold"])
    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Prediction Results")

    left, right, third = st.columns(3)

    left.metric(
        "Churn Probability",
        f"{probability:.1%}",
    )
    right.metric(
        "Decision Threshold",
        f"{threshold:.1%}",
    )
    third.metric(
        "Prediction",
        "Likely to Churn" if prediction else "Likely to Stay",
    )

    st.progress(probability)

    if prediction:
        st.warning("This customer exceeds the model's churn threshold.")
    else:
        st.success("This customer is below the model's churn threshold.")

st.divider()
st.caption(
    "Portfolio demonstration trained on public IBM Telco data. "
    "Predictions are not intended for real business decisions."
)
