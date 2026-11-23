# MLForge — Modeling and Evaluation Methodology

## Dataset

MLForge uses the IBM Telco Customer Churn dataset containing 7,043 observations and 19 predictive features after excluding the customer identifier and target.

The target variable is binary: churn or no churn.

## Data Preparation

- Validate required columns and target labels.
- Remove customer identifiers from predictors.
- Convert `TotalCharges` to a numeric feature.
- Impute missing numerical and categorical values.
- Scale numerical features.
- One-hot encode categorical features.
- Fit preprocessing transformations inside model pipelines.

## Baseline Modeling

Three algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

Model selection initially used stratified cross-validation ROC-AUC.

## Hyperparameter Optimization

Optuna was used to search candidate model configurations.

The Day 3 optimization used 10 trials and selected XGBoost.

The selected configuration included approximately:

| Hyperparameter | Value |
|---|---:|
| Estimators | 400 |
| Maximum depth | 3 |
| Learning rate | 0.013399 |
| Subsample | 0.90527 |

## Deployed Model Results

The deployed `v0.1.0` model reported:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8458 |
| PR-AUC | 0.6678 |
| Accuracy | 0.7764 |
| Precision | 0.5606 |
| Recall | 0.7299 |
| F1 | 0.6341 |
| Threshold | 0.33 |

These are experimental held-out results, not externally validated production performance.

## Decision Threshold Optimization

A probability threshold converts model scores into binary decisions.

Two strategies were evaluated:

**F1 optimization:** Select the threshold maximizing out-of-fold F1.

**Business-cost optimization:** Select the threshold minimizing the assumed cost of false positives and false negatives.

The illustrative cost function is:

\[
C = 20FP + 150FN
\]

The values are hypothetical and are not derived from actual retention campaigns.

## Champion–Challenger Evaluation

A subsequent experiment compared retrained XGBoost and Logistic Regression configurations using a shared train/test split.

Each model independently optimized its threshold using training-only out-of-fold predictions.

### Business-Cost Results

| Model | Threshold | Recall | Estimated cost |
|---|---:|---:|---:|
| XGBoost | 0.09 | 0.9545 | $13,350 |
| Logistic Regression | 0.29 | 0.9225 | $13,670 |

Under the assumed costs, XGBoost had the lower estimated classification cost.

The experiment is exploratory and does not establish real financial savings.

## Quality Gates

Candidate models are evaluated against minimum performance requirements and permitted regression limits.

A failed quality gate prevents the candidate from being considered acceptable under the configured policy.

Quality-gate results do not automatically trigger production deployment.

## Statistical Drift Monitoring

The monitoring demonstration uses deliberately shifted customer data.

Tests include:

- Kolmogorov–Smirnov for numerical distributions.
- Chi-square for categorical distributions.
- Benjamini–Hochberg correction for multiple comparisons.

Four of 19 features were flagged in the reported simulation.

Statistical significance does not directly measure the practical magnitude of drift or the effect on predictive performance.

## Methodological Limitations

Earlier experiments used different random splits, and later experiments were informed by previously examined results from the same public dataset.

Therefore, comparisons should not be presented as fully independent external validation.

The current workflow also does not measure calibration under real deployment conditions, retention intervention outcomes, or production performance degradation.

Future work should introduce immutable dataset partitions, independent external evaluation, calibration analysis, and labeled production feedback.