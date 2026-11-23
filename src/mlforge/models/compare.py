import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    average_precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from mlforge.data.loader import load_dataset
from mlforge.models.quality_gate import check_quality

EVALUATION_SEED = 2026


def evaluate(model, metadata, X, y):
    """Evaluate a fitted model with its saved threshold."""
    features = metadata["features"]
    threshold = float(metadata["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")

    if set(features) != set(X.columns):
        raise ValueError("Model feature schema mismatch")

    probabilities = model.predict_proba(X.reindex(columns=features))[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
    }


def load_bundle(directory):
    """Load a trusted model and its matching metadata."""
    directory = Path(directory)

    model = joblib.load(directory / "best_model.joblib")
    metadata = json.loads((directory / "metadata.json").read_text())

    return model, metadata


def main():
    champion, champion_meta = load_bundle("artifacts/champion")
    candidate, candidate_meta = load_bundle("artifacts/candidate")

    X, y = load_dataset()

    # Must match the original Day 3 training split.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=EVALUATION_SEED,
    )

    champion_metrics = evaluate(champion, champion_meta, X_test, y_test)
    candidate_metrics = evaluate(candidate, candidate_meta, X_test, y_test)

    result = check_quality(candidate_metrics, champion_metrics)

    report = {
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_samples": len(X_test),
        "champion": champion_metrics,
        "candidate": candidate_metrics,
        "checks": result.checks,
        "passed": result.passed,
        "reasons": result.reasons,
    }

    output = Path("artifacts/promotion")
    output.mkdir(parents=True, exist_ok=True)

    (output / "quality_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2), flush=True)

    if not result.passed:
        raise SystemExit("Candidate failed quality gates")


if __name__ == "__main__":
    main()
