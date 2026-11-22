import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)


def optimize_threshold(model, X, y):
    """Select threshold using out-of-fold predictions."""
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=123,
    )

    probabilities = cross_val_predict(
        clone(model),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    thresholds = np.linspace(0.1, 0.9, 81)

    scores = [
        f1_score(
            y,
            probabilities >= threshold,
            zero_division=0,
        )
        for threshold in thresholds
    ]

    best_index = int(np.argmax(scores))

    return {
        "threshold": float(thresholds[best_index]),
        "validation_f1": float(scores[best_index]),
    }
