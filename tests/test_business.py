import numpy as np
import pytest

from mlforge.models.business import calculate_business_cost
from mlforge.models.decision import select_threshold


def test_business_cost():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.2, 0.9])

    result = calculate_business_cost(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["total_cost"] == 170.0
    assert result["cost_per_customer"] == 42.5


def test_threshold_optimization():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    result = select_threshold(
        y_true,
        probabilities,
        strategy="business_cost",
    )

    assert 0.05 <= result["threshold"] <= 0.95
    assert result["validation_score"] == 0.0


def test_invalid_cost():
    with pytest.raises(ValueError):
        calculate_business_cost(
            [0, 1],
            [0.2, 0.8],
            threshold=0.5,
            false_negative_cost=-10,
        )
