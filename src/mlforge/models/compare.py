import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from mlforge.data.loader import load_dataset
from mlforge.evaluate import evaluate_binary_classifier
from mlforge.models.quality_gate import check_quality

EVALUATION_SEED = 2026
BOOTSTRAP_SAMPLES = 1000
OUTPUT_DIR = Path("artifacts/promotion")


def evaluate(model, metadata, X, y):
    """Evaluate a fitted model with its saved threshold."""
    features = metadata["features"]
    threshold = float(metadata["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")

    if len(features) != len(set(features)) or set(features) != set(X.columns):
        raise ValueError("Model feature schema mismatch")

    probabilities = model.predict_proba(X.reindex(columns=features))[:, 1]

    metrics = evaluate_binary_classifier(
        y_true=y,
        probabilities=probabilities,
        threshold=threshold,
    )

    return metrics, probabilities


def load_bundle(directory):
    """Load a trusted model and its matching metadata."""
    directory = Path(directory)

    model = joblib.load(directory / "best_model.joblib")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    return model, metadata


def paired_bootstrap(y, champion_prob, candidate_prob):
    """Estimate paired uncertainty in discrimination differences."""
    labels = np.asarray(y)
    champion_prob = np.asarray(champion_prob)
    candidate_prob = np.asarray(candidate_prob)

    if not (len(labels) == len(champion_prob) == len(candidate_prob)):
        raise ValueError("Prediction arrays must be aligned")

    if len(np.unique(labels)) != 2:
        raise ValueError("Both classes are required")

    rng = np.random.default_rng(EVALUATION_SEED)
    differences = {"roc_auc": [], "pr_auc": []}

    for _ in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, len(labels), size=len(labels))
        sample_y = labels[indices]

        if len(np.unique(sample_y)) < 2:
            continue

        champion = champion_prob[indices]
        candidate = candidate_prob[indices]

        differences["roc_auc"].append(
            float(
                roc_auc_score(sample_y, candidate) - roc_auc_score(sample_y, champion)
            )
        )
        differences["pr_auc"].append(
            float(
                average_precision_score(sample_y, candidate)
                - average_precision_score(sample_y, champion)
            )
        )

    if any(not values for values in differences.values()):
        raise ValueError("No valid bootstrap samples")

    return {
        metric: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "mean": float(np.mean(values)),
            "valid_resamples": len(values),
        }
        for metric, values in differences.items()
    }


def main():
    """Evaluate candidate against the saved champion."""
    champion, champion_meta = load_bundle("artifacts/champion")
    candidate, candidate_meta = load_bundle("artifacts/candidate")

    X, y = load_dataset()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=EVALUATION_SEED,
    )

    champion_metrics, champion_prob = evaluate(champion, champion_meta, X_test, y_test)
    candidate_metrics, candidate_prob = evaluate(
        candidate, candidate_meta, X_test, y_test
    )

    result = check_quality(
        candidate_metrics,
        champion_metrics,
    )

    intervals = paired_bootstrap(
        y_test,
        champion_prob,
        candidate_prob,
    )

    metric_differences = {
        key: float(candidate_metrics[key] - champion_metrics[key])
        for key in (
            "roc_auc",
            "pr_auc",
            "recall",
            "f1",
            "brier_score",
            "log_loss",
        )
    }

    report = {
        "comparison_type": "saved_model_promotion",
        "evaluation_status": "historical_holdout_reuse",
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_samples": len(X_test),
        "champion": champion_metrics,
        "candidate": candidate_metrics,
        "metric_differences": metric_differences,
        "paired_bootstrap_95ci": intervals,
        "champion_metadata": {
            "git_commit": champion_meta.get("git_commit"),
            "dataset_sha256": champion_meta.get("dataset_sha256"),
            "calibrated": champion_meta.get("calibrated", False),
        },
        "candidate_metadata": {
            "git_commit": candidate_meta.get("git_commit"),
            "dataset_sha256": candidate_meta.get("dataset_sha256"),
            "calibrated": candidate_meta.get("calibrated", False),
        },
        "checks": result.checks,
        "passed": result.passed,
        "reasons": result.reasons,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "quality_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2), flush=True)

    if not result.passed:
        raise SystemExit("Candidate failed quality gates")


if __name__ == "__main__":
    main()
