import numpy as np
from sklearn.metrics import f1_score

from mlforge.models.business import calculate_business_cost


def select_threshold(
    y_true,
    probabilities,
    strategy: str = "business_cost",
    false_positive_cost: float = 20.0,
    false_negative_cost: float = 150.0,
) -> dict:
    """Select a threshold using out-of-fold predictions."""
    if strategy not in {"business_cost", "f1"}:
        raise ValueError("Unknown threshold strategy")

    thresholds = np.linspace(0.05, 0.95, 91)
    scores = []

    for threshold in thresholds:
        if strategy == "f1":
            score = f1_score(
                y_true,
                np.asarray(probabilities) >= threshold,
                zero_division=0,
            )
        else:
            result = calculate_business_cost(
                y_true,
                probabilities,
                float(threshold),
                false_positive_cost,
                false_negative_cost,
            )
            score = -result["total_cost"]

        scores.append(float(score))

    best_index = int(np.argmax(scores))

    return {
        "strategy": strategy,
        "threshold": float(thresholds[best_index]),
        "validation_score": scores[best_index],
        "false_positive_cost": false_positive_cost,
        "false_negative_cost": false_negative_cost,
    }
