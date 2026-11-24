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

from mlforge.data.loader import download_dataset, load_dataset
from mlforge.data.provenance import save_provenance
from mlforge.models.explain import generate_explanations
from mlforge.models.optimization import optimize_models
from mlforge.models.threshold import optimize_threshold

OUTPUT_DIR = Path("artifacts/tuning")
RANDOM_STATE = 2026
N_TRIALS = 10


def main() -> None:
    """Run cloud-based model optimization and evaluation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 1. Download dataset and record provenance
    # -----------------------------------------------------

    dataset_path = download_dataset()

    provenance = save_provenance(
        dataset_path,
        OUTPUT_DIR / "provenance.json",
    )

    print(
        f"Dataset SHA-256: {provenance['dataset_sha256']}",
        flush=True,
    )

    print(
        f"Training Git commit: {provenance['git_commit']}",
        flush=True,
    )

    # -----------------------------------------------------
    # 2. Load and validate dataset
    # -----------------------------------------------------

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # -----------------------------------------------------
    # 3. Configure MLflow
    # -----------------------------------------------------

    tracking_db = (OUTPUT_DIR / "mlflow.db").resolve()

    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")

    mlflow.set_experiment("mlforge-optimization")

    # -----------------------------------------------------
    # 4. Hyperparameter optimization
    # -----------------------------------------------------

    model, study = optimize_models(
        X_train,
        y_train,
        n_trials=N_TRIALS,
    )

    # -----------------------------------------------------
    # 5. Optimize decision threshold
    # -----------------------------------------------------

    threshold_result = optimize_threshold(
        model,
        X_train,
        y_train,
    )

    threshold = threshold_result["threshold"]

    # -----------------------------------------------------
    # 6. Fit final model
    # -----------------------------------------------------

    model.fit(X_train, y_train)

    # -----------------------------------------------------
    # 7. Evaluate on held-out test data
    # -----------------------------------------------------

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(
            average_precision_score(
                y_test,
                probabilities,
            )
        ),
        "threshold": float(threshold),
        "cv_roc_auc": float(study.best_value),
        "validation_f1": float(threshold_result["validation_f1"]),
    }

    # -----------------------------------------------------
    # 8. Save experiment results
    # -----------------------------------------------------

    with mlflow.start_run(run_name="optimized-model"):
        mlflow.log_params(study.best_params)

        mlflow.log_params(
            {
                "random_state": RANDOM_STATE,
                "n_trials": N_TRIALS,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.set_tags(
            {
                "git_commit": provenance["git_commit"],
                "dataset_sha256": provenance["dataset_sha256"],
                "python_version": provenance["python_version"],
            }
        )

        # Save trained preprocessing + classifier pipeline.
        joblib.dump(
            model,
            OUTPUT_DIR / "best_model.joblib",
        )

        metadata = {
            "model": study.best_params["model_name"],
            "threshold": float(threshold),
            "best_params": study.best_params,
            "features": X.columns.tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "random_state": RANDOM_STATE,
            "dataset_sha256": provenance["dataset_sha256"],
            "git_commit": provenance["git_commit"],
        }

        (OUTPUT_DIR / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        (OUTPUT_DIR / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        # Save Optuna trial history.
        study.trials_dataframe().to_csv(
            OUTPUT_DIR / "optuna_trials.csv",
            index=False,
        )

        # -------------------------------------------------
        # 9. Generate evaluation plots
        # -------------------------------------------------

        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
        )

        plt.savefig(
            OUTPUT_DIR / "roc_curve.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        PrecisionRecallDisplay.from_predictions(
            y_test,
            probabilities,
        )

        plt.savefig(
            OUTPUT_DIR / "precision_recall.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
        )

        plt.savefig(
            OUTPUT_DIR / "confusion_matrix.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # -------------------------------------------------
        # 10. Generate SHAP explanations
        # -------------------------------------------------

        generate_explanations(
            model,
            X_test,
            OUTPUT_DIR,
        )

        # -------------------------------------------------
        # 11. Log experiment artifacts to MLflow
        # -------------------------------------------------

        artifact_files = [
            "metadata.json",
            "metrics.json",
            "provenance.json",
            "optuna_trials.csv",
            "roc_curve.png",
            "precision_recall.png",
            "confusion_matrix.png",
            "shap_summary.png",
        ]

        for filename in artifact_files:
            path = OUTPUT_DIR / filename

            if path.is_file():
                mlflow.log_artifact(
                    str(path),
                    artifact_path="evaluation",
                )

    # -----------------------------------------------------
    # 12. Print final results
    # -----------------------------------------------------

    print(
        json.dumps(
            {
                "best_model": study.best_params["model_name"],
                "dataset_sha256": provenance["dataset_sha256"],
                "git_commit": provenance["git_commit"],
                "metrics": metrics,
            },
            indent=2,
        ),
        flush=True,
    )

    print(
        "MLForge optimization and provenance tracking completed!",
        flush=True,
    )


if __name__ == "__main__":
    main()
