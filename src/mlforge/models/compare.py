import hashlib
import json
import os
import re
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

CHAMPION_DIR = Path("artifacts/champion")
CANDIDATE_DIR = Path("artifacts/candidate")


def evaluate(model, metadata, X, y):
    """Evaluate a fitted model using its saved decision threshold."""
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

    differences = {
        "roc_auc": [],
        "pr_auc": [],
    }

    for _ in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(
            0,
            len(labels),
            size=len(labels),
        )

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


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of an artifact."""
    with path.open("rb") as source:
        return hashlib.file_digest(
            source,
            "sha256",
        ).hexdigest()


def build_promotion_evidence(
    *,
    report: dict,
    champion_metadata: dict,
    candidate_metadata: dict,
) -> dict:
    """Bind a passed quality gate to the evaluated model artifacts."""
    training_run_id = os.environ["MLFORGE_CANDIDATE_RUN_ID"]
    evaluation_run_id = os.environ["GITHUB_RUN_ID"]
    champion_version = os.environ["MLFORGE_CHAMPION_VERSION"]

    if not training_run_id.isdecimal():
        raise ValueError("Invalid candidate training run ID")

    if not evaluation_run_id.isdecimal():
        raise ValueError("Invalid evaluation run ID")

    if not re.fullmatch(
        r"v[0-9]+\.[0-9]+\.[0-9]+",
        champion_version,
    ):
        raise ValueError("Invalid champion version")

    if report.get("passed") is not True:
        raise ValueError("Cannot generate positive evidence for a failed quality gate")

    candidate_sha256 = sha256_file(CANDIDATE_DIR / "best_model.joblib")

    champion_sha256 = sha256_file(CHAMPION_DIR / "best_model.joblib")

    return {
        "schema_version": 1,
        "candidate": {
            "model_sha256": candidate_sha256,
            "training_run_id": int(training_run_id),
            "training_commit": candidate_metadata["git_commit"],
            "dataset_sha256": candidate_metadata["dataset_sha256"],
        },
        "champion": {
            "version": champion_version,
            "model_sha256": champion_sha256,
            "training_commit": champion_metadata.get("git_commit"),
        },
        "evaluation": {
            "run_id": int(evaluation_run_id),
            "workflow": "MLForge - Model Quality Gate",
            "evaluation_status": report["evaluation_status"],
            "quality_gate_passed": True,
            "checks": report["checks"],
            "candidate_metrics": report["candidate"],
            "champion_metrics": report["champion"],
            "metric_differences": report["metric_differences"],
            "paired_bootstrap_95ci": report["paired_bootstrap_95ci"],
        },
    }


def main():
    """Evaluate candidate against champion and save audit evidence."""
    champion, champion_meta = load_bundle(CHAMPION_DIR)
    candidate, candidate_meta = load_bundle(CANDIDATE_DIR)

    X, y = load_dataset()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=EVALUATION_SEED,
    )

    champion_metrics, champion_prob = evaluate(
        champion,
        champion_meta,
        X_test,
        y_test,
    )

    candidate_metrics, candidate_prob = evaluate(
        candidate,
        candidate_meta,
        X_test,
        y_test,
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT_DIR / "quality_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    # Never retain stale positive evidence from a previous run.
    evidence_path = OUTPUT_DIR / "promotion_evidence.json"
    evidence_path.unlink(missing_ok=True)

    if not result.passed:
        print(json.dumps(report, indent=2), flush=True)
        raise SystemExit("Candidate failed quality gates")

    evidence = build_promotion_evidence(
        report=report,
        champion_metadata=champion_meta,
        candidate_metadata=candidate_meta,
    )

    evidence_path.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2), flush=True)
    print(
        "Candidate passed quality gates; promotion evidence generated.",
        flush=True,
    )


if __name__ == "__main__":
    main()
