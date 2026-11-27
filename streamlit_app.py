import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mlforge.deployment import download_model  # noqa: E402

MODEL_DIR = Path(os.getenv("MLFORGE_MODEL_DIR", "artifacts/tuning"))

MONITORING_URL = (
    "https://github.com/hamedhoney/MLForge/"
    "releases/download/monitoring-latest/"
    "drift_report.json"
)

st.set_page_config(
    page_title="MLForge | Customer Churn",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------


@st.cache_resource
def load_model():
    """Load the trusted model and its metadata."""
    model_path = MODEL_DIR / "best_model.joblib"
    metadata_path = MODEL_DIR / "metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        download_model()

    # Only load artifacts from the verified GitHub release.
    model = joblib.load(model_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return model, metadata


# ---------------------------------------------------------
# Monitoring report loading
# ---------------------------------------------------------


@st.cache_data(ttl=300)
def load_monitoring_report():
    """Retrieve the latest published monitoring report."""
    with urlopen(MONITORING_URL, timeout=20) as response:
        return json.load(response)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------


st.title("MLForge")
st.caption("End-to-End MLOps Platform | Customer Churn Prediction & Monitoring")

try:
    model, metadata = load_model()
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()

threshold = float(metadata["threshold"])

st.sidebar.header("Model Information")

st.sidebar.write(f"**Model:** {metadata['model']}")
st.sidebar.write(f"**Decision threshold:** {threshold:.2f}")
st.sidebar.write(f"**Input features:** {len(metadata['features'])}")
st.sidebar.write("**Model version:** v0.1.0")

st.sidebar.divider()

st.sidebar.markdown("[GitHub Repository](https://github.com/hamedhoney/MLForge)")

st.sidebar.markdown("[Model Releases](https://github.com/hamedhoney/MLForge/releases)")


# ---------------------------------------------------------
# Dashboard navigation
# ---------------------------------------------------------


prediction_tab, monitoring_tab = st.tabs(
    [
        "Customer Churn Prediction",
        "Model Monitoring",
    ]
)


# =========================================================
# TAB 1: CUSTOMER CHURN PREDICTION
# =========================================================


with prediction_tab:
    st.header("Customer Churn Prediction")

    st.write("Enter a customer profile to estimate the probability of customer churn.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        # -------------------------------------------------
        # Customer demographics
        # -------------------------------------------------

        with col1:
            st.subheader("Customer Information")

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"],
            )

            dependents = st.selectbox(
                "Dependents",
                ["No", "Yes"],
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                value=12,
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
            )

            multiple = st.selectbox(
                "Multiple Lines",
                [
                    "No",
                    "Yes",
                    "No phone service",
                ],
            )

        # -------------------------------------------------
        # Customer services
        # -------------------------------------------------

        with col2:
            st.subheader("Services")

            internet = st.selectbox(
                "Internet Service",
                [
                    "Fiber optic",
                    "DSL",
                    "No",
                ],
            )

            security = st.selectbox(
                "Online Security",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

            backup = st.selectbox(
                "Online Backup",
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
            )

            protection = st.selectbox(
                "Device Protection",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

            support = st.selectbox(
                "Tech Support",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

            tv = st.selectbox(
                "Streaming TV",
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
            )

            movies = st.selectbox(
                "Streaming Movies",
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
            )

        # -------------------------------------------------
        # Billing information
        # -------------------------------------------------

        with col3:
            st.subheader("Billing Information")

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year",
                ],
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
            )

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

    # -----------------------------------------------------
    # Run inference
    # -----------------------------------------------------

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

        try:
            start_time = time.perf_counter()

            probability = float(model.predict_proba(frame)[0, 1])

            latency_ms = (time.perf_counter() - start_time) * 1000

            prediction = int(probability >= threshold)

        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        st.divider()

        st.subheader("Prediction Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Churn Probability",
            f"{probability:.1%}",
        )

        result_col2.metric(
            "Decision Threshold",
            f"{threshold:.1%}",
        )

        result_col3.metric(
            "Prediction",
            ("Likely to Churn" if prediction else "Likely to Stay"),
        )

        st.progress(min(max(probability, 0.0), 1.0))

        if prediction:
            st.warning("This customer exceeds the model's churn threshold.")
        else:
            st.success("This customer is below the model's churn threshold.")

        st.subheader("Inference Performance")

        latency_col1, latency_col2 = st.columns(2)

        latency_col1.metric(
            "Prediction Latency",
            f"{latency_ms:.2f} ms",
        )

        latency_col2.metric(
            "Model Version",
            "v0.1.0",
        )

        with st.expander("View prediction details"):
            st.json(
                {
                    "model": metadata["model"],
                    "model_version": "v0.1.0",
                    "prediction": prediction,
                    "churn_probability": probability,
                    "threshold": threshold,
                    "latency_ms": latency_ms,
                }
            )

    st.divider()

    st.caption(
        "Portfolio demonstration trained on public "
        "IBM Telco Customer Churn data. "
        "Predictions are not intended for real "
        "business decisions."
    )


# =========================================================
# TAB 2: MODEL MONITORING
# =========================================================


with monitoring_tab:
    st.header("Model Monitoring")

    st.caption(
        "Statistical data drift detection using a "
        "reproducible simulated distribution shift. "
        "This is not live production monitoring."
    )

    if st.button(
        "Refresh Monitoring Report",
        use_container_width=False,
    ):
        load_monitoring_report.clear()
        st.rerun()

    try:
        monitoring = load_monitoring_report()

        st.subheader("Monitoring Overview")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Monitored Features",
            monitoring["total_features"],
        )

        metric_col2.metric(
            "Drifted Features",
            monitoring["drifted_feature_count"],
        )

        metric_col3.metric(
            "Drift Share",
            f"{monitoring['drift_share']:.1%}",
        )

        st.write(
            "**Report generated:**",
            monitoring["timestamp"],
        )

        st.write(
            "**Scenario:**",
            monitoring.get(
                "scenario",
                "Unknown",
            ),
        )

        st.write(
            "**Reference samples:**",
            monitoring.get(
                "reference_samples",
                "N/A",
            ),
        )

        st.write(
            "**Current samples:**",
            monitoring.get(
                "current_samples",
                "N/A",
            ),
        )

        if monitoring["dataset_drift_detected"]:
            st.warning("Distribution shift detected in simulated customer data.")
        else:
            st.success("No statistically significant distribution shift detected.")

        # -------------------------------------------------
        # Feature drift analysis
        # -------------------------------------------------

        st.divider()

        st.subheader("Feature Drift Analysis")

        feature_results = pd.DataFrame.from_dict(
            monitoring["features"],
            orient="index",
        )

        feature_results.index.name = "Feature"

        display_columns = [
            "test",
            "statistic",
            "p_value",
            "adjusted_p_value",
            "drift_detected",
            "reference_missing_pct",
            "current_missing_pct",
        ]

        available_columns = [
            column for column in display_columns if column in feature_results.columns
        ]

        st.dataframe(
            feature_results[available_columns],
            use_container_width=True,
        )

        # -------------------------------------------------
        # Statistical significance visualization
        # -------------------------------------------------

        st.divider()

        st.subheader("Feature Drift Statistical Significance")

        chart = feature_results.copy()

        chart["significance"] = -np.log10(chart["adjusted_p_value"].clip(lower=1e-15))

        chart = chart.sort_values(
            "significance",
            ascending=False,
        )

        st.bar_chart(
            chart["significance"],
        )

        st.caption(
            "The chart shows -log10(adjusted p-value). "
            "Higher values indicate stronger statistical "
            "evidence of distribution change, not "
            "necessarily greater practical impact."
        )

        # -------------------------------------------------
        # Drifted features
        # -------------------------------------------------

        st.divider()

        st.subheader("Detected Drift Features")

        drifted_features = monitoring.get(
            "drifted_features",
            [],
        )

        if drifted_features:
            for feature in drifted_features:
                st.write(f"• {feature}")
        else:
            st.info("No features were flagged for drift.")

        # -------------------------------------------------
        # Simulation details
        # -------------------------------------------------

        simulation = monitoring.get(
            "simulation",
            {},
        )

        if simulation:
            with st.expander("View simulated distribution changes"):
                st.json(simulation)

        # -------------------------------------------------
        # Raw report
        # -------------------------------------------------

        with st.expander("View complete monitoring report"):
            st.json(monitoring)

    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.info(
            "Monitoring results are not available yet. "
            "Run the MLForge monitoring workflow "
            "in GitHub Actions."
        )

        st.caption(f"Report loading details: {exc}")


# =========================================================
# FOOTER
# =========================================================


st.divider()

st.caption("MLForge | GitHub Actions · MLflow · Optuna · XGBoost · FastAPI · Streamlit")
