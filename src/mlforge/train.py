import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from mlforge.config import DEFAULT_CONFIG
from mlforge.data.loader import load_dataset
from mlforge.evaluate import evaluate_binary_classifier
from mlforge.models.pipeline import build_models

RANDOM_STATE = DEFAULT_CONFIG.random_state
OUTPUT_DIR = Path("artifacts")


def evaluate_model(model, X_test, y_test):
    """Evaluate a fitted model using the shared evaluator."""
    probabilities = model.predict_proba(X_test)[:, 1]

    return evaluate_binary_classifier(
        y_true=y_test,
        probabilities=probabilities,
        threshold=0.5,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DEFAULT_CONFIG.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    tracking_db = (OUTPUT_DIR / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("mlforge-telco-churn")

    models = build_models(X_train)

    cv = StratifiedKFold(
        n_splits=DEFAULT_CONFIG.cv_folds,
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
