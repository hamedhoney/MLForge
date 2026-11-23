import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from mlforge.data.loader import load_dataset
from mlforge.models.pipeline import build_models
from mlforge.models.quality_gate import check_quality

OUTPUT_DIR = Path("artifacts/challenger")
RANDOM_STATE = 2027


def evaluate(model, X, y, threshold):
    """Evaluate a fitted model using a fixed threshold."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
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

    models = build_models(X_train)

    champion = models["xgboost"]
    champion.set_params(
        classifier__n_estimators=400,
        classifier__max_depth=3,
        classifier__learning_rate=0.013399,
        classifier__subsample=0.90527,
    )

    challenger = models["logistic_regression"]
    challenger.set_params(classifier__C=1.0)

    champion.fit(X_train, y_train)
    challenger.fit(X_train, y_train)

    # Initial comparison uses a common fixed threshold.
    # We will tune thresholds on validation data later.
    threshold = 0.33

    champion_metrics = evaluate(champion, X_test, y_test, threshold)
    challenger_metrics = evaluate(challenger, X_test, y_test, threshold)

    result = check_quality(challenger_metrics, champion_metrics)

    report = {
        "comparison_type": "retrained_configuration_comparison",
        "random_state": RANDOM_STATE,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "champion_configuration": "xgboost",
        "challenger_configuration": "logistic_regression",
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "quality_gate": {
            "passed": result.passed,
            "checks": result.checks,
            "reasons": result.reasons,
        },
    }

    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(report, indent=2))

    joblib.dump(
        challenger,
        OUTPUT_DIR / "challenger_model.joblib",
    )

    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
