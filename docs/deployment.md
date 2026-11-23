# MLForge — Cloud Training and Deployment Guide

## Overview

MLForge uses GitHub Actions for training and testing, GitHub Releases for versioned model storage, and Streamlit Community Cloud for interactive inference.

Local model execution is not required.

## Prerequisites

- GitHub account
- Public MLForge repository
- GitHub Actions enabled
- Streamlit Community Cloud account
- Python project managed using uv

## 1. Cloud Training

Open the GitHub repository's Actions tab.

Select **MLForge - Cloud Training** and start the workflow.

The workflow installs dependencies, downloads the Kaggle dataset, runs validation and tests, trains models, and uploads artifacts.

## 2. Hyperparameter Optimization

The optimization job runs Optuna and threshold selection in the cloud.

Generated outputs include:

```text
best_model.joblib
metadata.json
metrics.json
optuna_trials.csv
roc_curve.png
precision_recall.png
confusion_matrix.png
shap_summary.png
mlflow.db
```

The SHAP image is generated when the selected model is supported by the explanation implementation.

## 3. Publish an Approved Model

Run **MLForge - Publish Model**.

Provide the ID of a trusted, successful optimization workflow and a new release version.

The release workflow downloads the model, generates SHA-256 checksums, and publishes the files through GitHub Releases.

Only trusted training artifacts should be loaded with joblib because pickle-based serialization can execute code during deserialization.

## 4. Deploy Streamlit

Open Streamlit Community Cloud and create an application using:

- Repository: `hamedhoney/MLForge`
- Branch: `main`
- Entry point: `streamlit_app.py`
- Python: 3.12 where available

Configure the following secrets using the approved release checksums:

```toml
MLFORGE_MODEL_SHA256 = "YOUR_APPROVED_MODEL_SHA256"
MLFORGE_METADATA_SHA256 = "YOUR_APPROVED_METADATA_SHA256"
```

The deployment module downloads and verifies model artifacts from the pinned release.

The deployed application currently uses release `v0.1.0`.

## 5. Live Application

[Open MLForge](https://mlforge-new.streamlit.app/)

The application provides customer churn predictions and a simulated monitoring dashboard.

## 6. Monitoring

Run **MLForge - Model Monitoring** from GitHub Actions.

The workflow generates a statistical drift report and publishes it to the `monitoring-latest` GitHub Release.

Open the Streamlit monitoring tab and refresh the report to display the latest results.

This is a reproducible simulated drift demonstration, not monitoring of real production traffic.

## 7. Champion–Challenger Evaluation

Run **MLForge - Challenger Evaluation**.

The workflow trains alternative model configurations and produces evaluation reports.

Use **MLForge - Model Quality Gate** for comparing approved candidate artifacts against the current champion under the configured evaluation policy.

The quality-gate workflow does not automatically deploy a candidate.

## 8. Updating the Production Model

The current application pins release `v0.1.0`.

Publishing a newer release alone does not update the deployed model.

To deploy a different version, update the release reference in `src/mlforge/deployment.py`, configure the corresponding checksums, and redeploy.

The model and metadata must always originate from the same training run.

## 9. Troubleshooting

**Model artifact missing:** Confirm the GitHub Release exists and contains both required model files.

**Checksum mismatch:** Confirm the configured checksums match the pinned release. Do not disable checksum verification.

**Dependency installation failure:** Check Python compatibility and the project dependency definitions.

**Monitoring report unavailable:** Run the monitoring workflow and verify the `monitoring-latest` release.

**Prediction schema mismatch:** Confirm that the deployed model's feature metadata matches the Streamlit input schema.

## 10. Free Cloud Limitations

GitHub-hosted runners are temporary and do not provide continuous application hosting.

GitHub Actions artifacts have retention limits.

Streamlit Community Cloud has resource and availability limits, and applications may sleep when inactive.

MLForge uses these services for a portfolio demonstration rather than guaranteed production uptime.