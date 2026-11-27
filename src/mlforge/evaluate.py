
"""Reusable binary classification evaluation utilities."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate probabilities at a predefined decision threshold."""
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)

    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be 1D and aligned.")

    if len(y) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary: 0 or 1.")

    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1].")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    predictions = (p >= threshold).astype(int)
    has_both_classes = len(np.unique(y)) == 2

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y, p)) if has_both_classes else None
        ),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
        "threshold": float(threshold),
        "n_samples": int(len(y)),
    }


def bootstrap_confidence_intervals(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 2026,
) -> dict[str, dict[str, float]]:
    """Compute percentile bootstrap intervals for ranking metrics."""
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)

    # Reuse input validation.
    evaluate_binary_classifier(y, p)

    if len(np.unique(y)) != 2:
        raise ValueError("Both classes are required for bootstrap AUC.")

    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    scores: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
    }

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y), size=len(y))
        sample_y = y[indices]
        sample_p = p[indices]

        if len(np.unique(sample_y)) < 2:
            continue

        scores["roc_auc"].append(
            float(roc_auc_score(sample_y, sample_p))
        )
        scores["pr_auc"].append(
            float(average_precision_score(sample_y, sample_p))
        )

    alpha = (1 - confidence) / 2

    if any(not values for values in scores.values()):
        raise ValueError("No valid bootstrap samples were generated.")

    return {
        metric: {
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1 - alpha)),
            "confidence": float(confidence),
            "valid_resamples": len(values),
        }
        for metric, values in scores.items()
    }
