import numpy as np
import pytest

from mlforge.evaluate import (
    bootstrap_confidence_intervals,
    evaluate_binary_classifier,
)


@pytest.fixture
def predictions():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    return y, p


def test_perfect_classification(predictions):
    y, p = predictions
    metrics = evaluate_binary_classifier(y, p)

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[4, 0], [0, 4]]


def test_invalid_probabilities():
    with pytest.raises(ValueError):
        evaluate_binary_classifier([0, 1], [0.2, 1.5])


def test_invalid_threshold():
    with pytest.raises(ValueError):
        evaluate_binary_classifier([0, 1], [0.2, 0.8], threshold=1.2)


def test_single_class_auc_is_undefined():
    metrics = evaluate_binary_classifier([0, 0], [0.1, 0.2])
    assert metrics["roc_auc"] is None


def test_bootstrap_reproducibility(predictions):
    y, p = predictions

    first = bootstrap_confidence_intervals(
        y, p, n_bootstrap=100, random_state=42
    )
    second = bootstrap_confidence_intervals(
        y, p, n_bootstrap=100, random_state=42
    )

    assert first == second


def test_bootstrap_bounds(predictions):
    y, p = predictions

    intervals = bootstrap_confidence_intervals(
        y, p, n_bootstrap=100
    )

    for interval in intervals.values():
        assert 0 <= interval["lower"] <= interval["upper"] <= 1
