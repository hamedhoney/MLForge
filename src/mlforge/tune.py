import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from mlforge.config import DEFAULT_CONFIG
from mlforge.data.loader import download_dataset, load_dataset
from mlforge.data.provenance import save_provenance
from mlforge.evaluate import (
    bootstrap_confidence_intervals,
    evaluate_binary_classifier,
)
from mlforge.models.explain import generate_explanations
from mlforge.models.optimization import optimize_models
from mlforge.models.threshold import optimize_threshold

OUTPUT_DIR = Path("artifacts/tuning")
RANDOM_STATE = DEFAULT_CONFIG.random_state
N_TRIALS = 10


def main() -> None:
    """Run model optimization and calibrated evaluation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset and provenance
    dataset_path = download_dataset()
    provenance = save_provenance(
        dataset_path,
        OUTPUT_DIR / "provenance.json",
    )

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DEFAULT_CONFIG.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # 2. Configure MLflow
    tracking_db = (OUTPUT_DIR / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("mlforge-optimization")

    # 3. Optuna optimization on training data
    model, study = optimize_models(
        X_train,
        y_train,
        n_trials=N_TRIALS,
    )

    # 4. Calibrate model probabilities
    calibration_cv = StratifiedKFold(
        n_splits=DEFAULT_CONFIG.cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=calibration_cv,
    )

    # 5. Optimize threshold using OOF calibrated probabilities
    threshold_result = optimize_threshold(
        calibrated_model,
        X_train,
        y_train,
    )
    threshold = threshold_result["threshold"]

    # 6. Fit final calibrated model
    calibrated_model.fit(X_train, y_train)

    # 7. Evaluate held-out data
    probabilities = calibrated_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = evaluate_binary_classifier(
        y_true=y_test,
        probabilities=probabilities,
        threshold=threshold,
    )

    metrics["cv_roc_auc"] = float(study.best_value)
    metrics["validation_f1"] = float(threshold_result["validation_f1"])

    confidence_intervals = bootstrap_confidence_intervals(
        y_true=y_test,
        probabilities=probabilities,
        n_bootstrap=DEFAULT_CONFIG.bootstrap_samples,
        confidence=DEFAULT_CONFIG.confidence_level,
        random_state=RANDOM_STATE,
    )

    # 8. Save experiment results
    with mlflow.start_run(run_name="optimized-calibrated-model"):
        mlflow.log_params(study.best_params)

        mlflow.log_params(
            {
                "random_state": RANDOM_STATE,
                "n_trials": N_TRIALS,
                "calibration_method": "sigmoid",
                "calibration_folds": DEFAULT_CONFIG.cv_folds,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        )

        # MLflow accepts scalar numeric metrics.
        mlflow.log_metrics(
            {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and value is not None
            }
        )

        mlflow.set_tags(
            {
                "git_commit": provenance["git_commit"],
                "dataset_sha256": provenance["dataset_sha256"],
                "python_version": provenance["python_version"],
            }
        )

        # Save calibrated model
        joblib.dump(
            calibrated_model,
            OUTPUT_DIR / "best_model.joblib",
        )

        metadata = {
            "model": study.best_params["model_name"],
            "calibrated": True,
            "calibration_method": "sigmoid",
            "threshold": threshold,
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

        (OUTPUT_DIR / "confidence_intervals.json").write_text(
            json.dumps(confidence_intervals, indent=2),
            encoding="utf-8",
        )

        study.trials_dataframe().to_csv(
            OUTPUT_DIR / "optuna_trials.csv",
            index=False,
        )

        # 9. Evaluation plots
        RocCurveDisplay.from_predictions(y_test, probabilities)
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

        # 10. SHAP explanations
        # Use the underlying fitted pipeline for compatibility
        # with the existing explanation implementation.
        fitted_model = calibrated_model.calibrated_classifiers_[0].estimator

        generate_explanations(
            fitted_model,
            X_test,
            OUTPUT_DIR,
        )

        # 11. Log artifacts
        artifact_files = [
            "metadata.json",
            "metrics.json",
            "confidence_intervals.json",
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

    print(
        json.dumps(
            {
                "best_model": study.best_params["model_name"],
                "dataset_sha256": provenance["dataset_sha256"],
                "git_commit": provenance["git_commit"],
                "metrics": metrics,
                "confidence_intervals": confidence_intervals,
            },
            indent=2,
        ),
        flush=True,
    )

    print("MLForge calibrated optimization completed!", flush=True)


if __name__ == "__main__":
    main()
