import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlforge.evaluate import evaluate_binary_classifier


def test_calibrated_probabilities_are_valid():
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=1000),
        method="sigmoid",
        cv=3,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))

    metrics = evaluate_binary_classifier(y_test, probabilities)

    assert 0 <= metrics["brier_score"] <= 1
    assert metrics["log_loss"] >= 0
