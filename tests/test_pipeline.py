"""Tests for the MLForge preprocessing pipeline."""

import pandas as pd

from mlforge.models.pipeline import build_models


def test_models_are_created():
    X = pd.DataFrame(
        {
            "tenure": [1, 12, 24, 36],
            "MonthlyCharges": [30.0, 50.0, 70.0, 90.0],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
            ],
        }
    )

    models = build_models(X)

    assert set(models) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }

    for model in models.values():
        assert "preprocessor" in model.named_steps
        assert "classifier" in model.named_steps
