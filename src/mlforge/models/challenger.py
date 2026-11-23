import json
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from mlforge.data.loader import load_dataset
from mlforge.models.business import calculate_business_cost
from mlforge.models.decision import select_threshold
from mlforge.models.pipeline import build_models
from mlforge.models.quality_gate import check_quality

OUTPUT_DIR = Path("artifacts/challenger")
RANDOM_STATE = 2027


def evaluate(model, X, y, threshold):
    """Evaluate a fitted model on common test records."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
    }

    costs = calculate_business_cost(
        y,
        probabilities,
        threshold,
    )

    return {**metrics, **costs}


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

    # Previously selected XGBoost configuration.
    # This is a retrained configuration, not deployed v0.1.0.
    models["xgboost"].set_params(
        classifier__n_estimators=400,
        classifier__max_depth=3,
        classifier__learning_rate=0.013399,
        classifier__subsample=0.90527,
    )

    models["logistic_regression"].set_params(
        classifier__C=1.0,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}
    fitted_models = {}

    for name, model in models.items():
        if name not in {"xgboost", "logistic_regression"}:
            continue

        print(f"Evaluating {name}...", flush=True)

        # Out-of-fold probabilities: each prediction comes
        # from a model that did not train on that row.
        oof_probabilities = cross_val_predict(
            clone(model),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        thresholds = {}

        for strategy in ("f1", "business_cost"):
            thresholds[strategy] = select_threshold(
                y_train,
                oof_probabilities,
                strategy=strategy,
            )

        # Fit once on the complete training partition.
        model.fit(X_train, y_train)
        fitted_models[name] = model

        evaluations = {}

        for strategy, selection in thresholds.items():
            evaluations[strategy] = evaluate(
                model,
                X_test,
                y_test,
                selection["threshold"],
            )

        results[name] = {
            "thresholds": thresholds,
            "test_metrics": evaluations,
        }

    champion = results["xgboost"]["test_metrics"]["business_cost"]
    challenger = results["logistic_regression"]["test_metrics"]["business_cost"]

    gate = check_quality(challenger, champion)

    report = {
        "comparison_type": "retrained_configuration_comparison",
        "evaluation_status": "exploratory",
        "random_state": RANDOM_STATE,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "business_cost_assumptions": {
            "false_positive_usd": 20,
            "false_negative_usd": 150,
        },
        "models": results,
        "quality_gate": {
            "passed": gate.passed,
            "checks": gate.checks,
            "reasons": gate.reasons,
        },
        "business_cost_difference_usd": (
            challenger["total_cost"] - champion["total_cost"]
        ),
    }

    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    for name, model in fitted_models.items():
        joblib.dump(
            model,
            OUTPUT_DIR / f"{name}_model.joblib",
        )

    print(json.dumps(report, indent=2), flush=True)
    print("Fair threshold comparison completed!", flush=True)


if __name__ == "__main__":
    main()
