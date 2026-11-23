# MLForge — System Architecture

## Overview

MLForge separates machine learning development into six layers: data ingestion, training, experiment tracking, model governance, inference, and monitoring.

## 1. Data Ingestion

**Source:** IBM Telco Customer Churn via KaggleHub.

The data loader downloads the public dataset on the GitHub-hosted runner, validates required columns, checks customer identifiers, converts `TotalCharges` to numeric, and separates features from labels.

The current implementation downloads the latest available dataset version. Pinning the dataset version and recording its checksum are recommended reproducibility improvements.

## 2. Training and Preprocessing

Training uses scikit-learn Pipelines and ColumnTransformer.

Numerical features receive median imputation and standard scaling. Categorical features receive most-frequent imputation and one-hot encoding.

Models include Logistic Regression, Random Forest, and XGBoost.

All preprocessing is fitted within the training pipeline to prevent leakage between cross-validation folds.

## 3. Experiment Tracking

MLflow records model parameters, performance metrics, and experiment metadata.

GitHub Actions uses a temporary SQLite tracking database. The database and generated reports are uploaded as workflow artifacts.

Because runners are ephemeral, these SQLite databases are individual experiment records rather than a persistent centralized tracking server.

## 4. Model Optimization

Optuna searches model hyperparameters using cross-validation.

Decision thresholds are optimized using out-of-fold training predictions.

The final fitted pipeline includes both preprocessing and classification components, allowing consistent inference.

## 5. Model Governance

The model quality-gate workflow compares champion and candidate models on identical evaluation records.

Quality checks include minimum ROC-AUC, PR-AUC, recall, and maximum permitted performance regression.

Passing the gate produces a report but does not automatically replace the deployed model.

The initial quality-gate demonstration compared the champion against the same model artifact. Later experiments compared retrained model configurations.

## 6. Model Storage and Deployment

Approved model artifacts are published through versioned GitHub Releases.

Each release contains the fitted model, metadata, and SHA-256 checksums.

Streamlit downloads and verifies the approved artifacts before loading them for inference.

The deployed application currently uses model release `v0.1.0`.

A separate FastAPI service is validated through Docker-based integration checks in GitHub Actions. The public Streamlit application runs independently of that container.

## 7. Monitoring

The monitoring workflow creates a reproducible synthetic distribution shift from the public customer dataset.

Numerical features are tested using the Kolmogorov–Smirnov test, and categorical features using chi-square tests.

Benjamini–Hochberg correction is applied across feature-level tests.

The monitoring report is published as a JSON release artifact and displayed in Streamlit.

The current monitoring system does not collect real production traffic or actual customer outcomes.

## 8. Security and Reliability

Implemented controls include typed API request validation, model artifact checksum verification, automated tests, and restricted GitHub Actions permissions.

Additional production requirements include authenticated prediction endpoints, artifact provenance verification, input rate limiting, dependency vulnerability scanning, and controlled release promotion.

## Architecture Principles

MLForge is designed around reproducibility, separation of training and inference, explicit model evaluation, measurable decision policies, and traceable model releases.