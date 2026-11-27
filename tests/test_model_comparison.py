import numpy as np
import pytest

from mlforge.models.compare import paired_bootstrap


def test_identical_models_have_zero_difference():
    y = np.array([0, 1] * 50)
    p = np.linspace(0.1, 0.9, len(y))

    result = paired_bootstrap(y, p, p)

    for metric in ("roc_auc", "pr_auc"):
        assert result[metric]["lower"] == pytest.approx(0.0)
        assert result[metric]["upper"] == pytest.approx(0.0)


def test_paired_bootstrap_is_reproducible():
    y = np.array([0, 1] * 50)
    champion = np.linspace(0.2, 0.8, len(y))
    candidate = np.linspace(0.1, 0.9, len(y))

    first = paired_bootstrap(y, champion, candidate)
    second = paired_bootstrap(y, champion, candidate)

    assert first == second


def test_invalid_array_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap(
            [0, 1, 0, 1],
            [0.2, 0.8],
            [0.1, 0.9, 0.2, 0.8],
        )
