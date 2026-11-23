import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_business_cost(
    y_true,
    probabilities,
    threshold: float,
    false_positive_cost: float = 20.0,
    false_negative_cost: float = 150.0,
) -> dict:
    """Calculate the cost of classification decisions."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("Costs cannot be negative")

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if y_true.ndim != 1 or probabilities.shape != y_true.shape:
        raise ValueError("Inputs must be equal-length 1D arrays")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Targets must be binary")

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probabilities must be finite and in [0, 1]")

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    total_cost = fp * false_positive_cost + fn * false_negative_cost

    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "total_cost": float(total_cost),
        "cost_per_customer": float(total_cost / len(y_true)),
    }
