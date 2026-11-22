"""Train, evaluate, and track churn prediction models."""

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from mlforge.data.loader import load_dataset
from mlforge.models.pipeline import build_models

RANDOM_STATE = 42
OUTPUT_DIR = Path("artifacts")


def evaluate_model(model, X_test, y_test):
    """Evaluate predictions on held-out test data."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    tracking_db = (OUTPUT_DIR / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("mlforge-telco-churn")

    models = build_models(X_train)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}
    best_model = None
    best_name = None
    best_cv_auc = -1.0

    for name, model in models.items():
        print(f"Training {name}...", flush=True)

        with mlflow.start_run(run_name=name):
            cv_scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "roc_auc": "roc_auc",
                    "f1": "f1",
                },
                n_jobs=1,
            )

            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)

            cv_auc = float(np.mean(cv_scores["test_roc_auc"]))
            cv_std = float(np.std(cv_scores["test_roc_auc"]))

            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("training_samples", len(X_train))

            mlflow.log_metrics(
                {
                    "cv_roc_auc_mean": cv_auc,
                    "cv_roc_auc_std": cv_std,
                    "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, float)
                    },
                }
            )

            results[name] = {
                "cv_roc_auc_mean": cv_auc,
                "cv_roc_auc_std": cv_std,
                "test_metrics": metrics,
            }

            if cv_auc > best_cv_auc:
                best_cv_auc = cv_auc
                best_model = model
                best_name = name

            print(f"{name}: {results[name]}", flush=True)

    # Select using training-set CV, not the held-out test.
    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")

    metadata = {
        "best_model": best_name,
        "selection_metric": "cv_roc_auc_mean",
        "best_cv_roc_auc": best_cv_auc,
        "random_state": RANDOM_STATE,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X.columns.tolist(),
    }

    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    print(f"Best model: {best_name}", flush=True)
    print("Cloud training completed successfully!", flush=True)


if __name__ == "__main__":
    main()
