from mlforge.models.quality_gate import check_quality


def test_good_candidate_passes():
    champion = {
        "roc_auc": 0.84,
        "pr_auc": 0.66,
        "recall": 0.70,
    }

    candidate = {
        "roc_auc": 0.85,
        "pr_auc": 0.67,
        "recall": 0.73,
    }

    result = check_quality(candidate, champion)

    assert result.passed
    assert not result.reasons


def test_regression_is_rejected():
    champion = {
        "roc_auc": 0.85,
        "pr_auc": 0.67,
        "recall": 0.75,
    }

    candidate = {
        "roc_auc": 0.82,
        "pr_auc": 0.63,
        "recall": 0.70,
    }

    result = check_quality(candidate, champion)

    assert not result.passed
    assert "roc_auc_regression" in result.reasons


def test_low_recall_is_rejected():
    champion = {
        "roc_auc": 0.84,
        "pr_auc": 0.66,
        "recall": 0.72,
    }

    candidate = {
        "roc_auc": 0.85,
        "pr_auc": 0.67,
        "recall": 0.50,
    }

    result = check_quality(candidate, champion)

    assert not result.passed
    assert "minimum_recall" in result.reasons
