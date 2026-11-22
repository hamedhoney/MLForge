"""Train and evaluate baseline ML models."""

import json
from pathlib import Path

import joblib
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def main() -> None:
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial smoke-test dataset.
    # We'll replace this with real churn data next.
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=2,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
        ),
    }

    mlflow.set_tracking_uri((output_dir / "mlruns").resolve().as_uri())
    mlflow.set_experiment("mlforge-baseline")

    results = {}
    best_auc = -1.0

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "f1": float(f1_score(y_test, predictions)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
            }

            mlflow.log_metrics(metrics)
            mlflow.log_param("model_name", name)

            results[name] = metrics

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                joblib.dump(
                    model,
                    output_dir / "best_model.joblib",
                )

            print(f"{name}: {metrics}")

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
