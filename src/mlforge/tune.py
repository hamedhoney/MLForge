import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from mlforge.data.loader import load_dataset
from mlforge.models.explain import generate_explanations
from mlforge.models.optimization import optimize_models
from mlforge.models.threshold import optimize_threshold

OUTPUT_DIR = Path("artifacts/tuning")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=2026,
    )

    tracking_db = (OUTPUT_DIR / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("mlforge-optimization")

    model, study = optimize_models(
        X_train,
        y_train,
        n_trials=10,
    )

    threshold_result = optimize_threshold(
        model,
        X_train,
        y_train,
    )
    threshold = threshold_result["threshold"]

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "threshold": threshold,
        "cv_roc_auc": float(study.best_value),
        "validation_f1": threshold_result["validation_f1"],
    }

    with mlflow.start_run(run_name="optimized-model"):
        mlflow.log_params(study.best_params)
        mlflow.log_metrics(metrics)

        joblib.dump(model, OUTPUT_DIR / "best_model.joblib")

        metadata = {
            "model": study.best_params["model_name"],
            "threshold": threshold,
            "best_params": study.best_params,
            "features": X.columns.tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

        trials = study.trials_dataframe()
        trials.to_csv(
            OUTPUT_DIR / "optuna_trials.csv",
            index=False,
        )

        RocCurveDisplay.from_predictions(y_test, probabilities)
        plt.savefig(
            OUTPUT_DIR / "roc_curve.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_test, probabilities)
        plt.savefig(
            OUTPUT_DIR / "precision_recall.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        plt.savefig(
            OUTPUT_DIR / "confusion_matrix.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        generate_explanations(
            model,
            X_test,
            OUTPUT_DIR,
        )

        for filename in [
            "metadata.json",
            "metrics.json",
            "optuna_trials.csv",
            "roc_curve.png",
            "precision_recall.png",
            "confusion_matrix.png",
            "shap_summary.png",
        ]:
            path = OUTPUT_DIR / filename
            if path.exists():
                mlflow.log_artifact(
                    str(path),
                    artifact_path="evaluation",
                )

    print(
        json.dumps(
            {
                "best_model": study.best_params["model_name"],
                "metrics": metrics,
            },
            indent=2,
        ),
        flush=True,
    )
    print("Day 3 optimization completed!", flush=True)


if __name__ == "__main__":
    main()
