import json
import os
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st

from api_client import (
    APIClientError,
    get_history,
    get_model_info,
    get_summary,
    predict_customer,
)

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "")).rstrip("/")
MONITORING_TOKEN = st.secrets.get("MONITORING_TOKEN", os.getenv("MONITORING_TOKEN", ""))

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
# Model information from the deployed inference API
# ---------------------------------------------------------


@st.cache_data(ttl=120)
def load_model_info():
    return get_model_info(API_URL)


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

if not API_URL:
    st.error("Configure API_URL in Streamlit Secrets.")
    st.stop()

try:
    info = load_model_info()
except APIClientError as exc:
    st.error(f"Inference API unavailable: {exc}")
    st.stop()

threshold = float(info["threshold"])
model_version = info.get("model_version") or "unknown"

st.sidebar.header("Model Information")
st.sidebar.write(f"**Model:** {info['model_name']}")
st.sidebar.write(f"**Decision threshold:** {threshold:.2f}")
st.sidebar.write(f"**Input features:** {info['feature_count']}")
st.sidebar.write(f"**Model version:** {model_version}")

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

        try:
            result, latency_ms = predict_customer(API_URL, customer)
            probability = float(result["churn_probability"])
            prediction = int(result["prediction"])
            threshold = float(result["threshold"])
        except (APIClientError, ValueError, KeyError, TypeError) as exc:
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
            model_version,
        )

        with st.expander("View prediction details"):
            st.json(
                {
                    "model": result["model_name"],
                    "model_version": model_version,
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
    st.subheader("Observed inference telemetry")
    st.caption(
        "Historical aggregates from the deployed API and Neon PostgreSQL; not ground-truth churn outcomes."
    )
    days = st.selectbox("Observation window", [7, 30, 90], index=1)
    if st.button("Refresh live telemetry"):
        st.rerun()
    if not MONITORING_TOKEN:
        st.warning("MONITORING_TOKEN is missing from Streamlit Secrets.")
    else:
        try:
            summary = get_summary(API_URL, MONITORING_TOKEN, days=days)
            source = summary.get("source", "unknown")
            if source != "persistent_database":
                st.warning(
                    "Showing process-local fallback metrics; persistent storage is unavailable."
                )
            else:
                st.success("Historical telemetry loaded from Neon PostgreSQL.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predictions", summary.get("total_predictions", 0))
            latency = summary.get("mean_latency_ms")
            m2.metric(
                "Mean latency", f"{latency:.1f} ms" if latency is not None else "N/A"
            )
            probability = summary.get("mean_probability")
            m3.metric(
                "Mean churn probability",
                f"{probability:.1%}" if probability is not None else "N/A",
            )
            churn = summary.get("predicted_churn_rate")
            m4.metric(
                "Predicted churn share", f"{churn:.1%}" if churn is not None else "N/A"
            )
            with st.expander("View telemetry summary"):
                st.json(summary)
        except APIClientError as exc:
            st.warning(f"Live telemetry unavailable: {exc}")

    st.subheader("Observed telemetry trends")
    version_filter = st.text_input(
        "Model version filter (optional)", value="", placeholder="e.g. v0.3.1"
    ).strip()
    if MONITORING_TOKEN:
        try:
            history = get_history(
                API_URL,
                MONITORING_TOKEN,
                days=days,
                model_version=version_filter or None,
            )
            points = pd.DataFrame(history["points"])
            if points.empty:
                st.info("No recorded predictions for this window and model version.")
            else:
                points["date"] = pd.to_datetime(points["date"], utc=True)
                points = points.set_index("date")
                st.caption(
                    "Daily UTC aggregates of observed API predictions; no customer profiles."
                )
                st.subheader("Prediction volume")
                st.bar_chart(points[["predictions"]])
                st.subheader("Inference latency (milliseconds)")
                st.line_chart(points[["mean_latency_ms", "p95_latency_ms"]])
                st.subheader("Prediction distribution")
                st.line_chart(points[["mean_probability", "predicted_churn_rate"]])
        except (APIClientError, KeyError, TypeError, ValueError) as exc:
            st.warning(f"Historical charts unavailable: {exc}")

    st.divider()
    st.subheader("Simulated feature drift demonstration")
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
