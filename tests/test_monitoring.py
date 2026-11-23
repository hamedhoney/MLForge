import numpy as np
import pandas as pd
import pytest

from mlforge.monitoring.drift import detect_drift


def test_identical_constant_data_has_no_drift():
    reference = pd.DataFrame(
        {
            "tenure": [12] * 100,
            "Contract": ["One year"] * 100,
        }
    )

    result = detect_drift(reference, reference.copy())

    assert result["dataset_drift_detected"] is False
    assert result["drifted_feature_count"] == 0


def test_detects_large_numerical_shift():
    rng = np.random.default_rng(42)

    reference = pd.DataFrame({"charges": rng.normal(50, 5, 500)})
    current = pd.DataFrame({"charges": rng.normal(100, 5, 500)})

    result = detect_drift(reference, current)

    assert result["features"]["charges"]["drift_detected"]


def test_rejects_schema_mismatch():
    reference = pd.DataFrame({"tenure": [1, 2, 3]})
    current = pd.DataFrame({"charges": [1, 2, 3]})

    with pytest.raises(ValueError):
        detect_drift(reference, current)
