from dataclasses import dataclass

MIN_ROC_AUC = 0.80
MIN_PR_AUC = 0.60
MIN_RECALL = 0.65
MAX_REGRESSION = 0.01


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: dict[str, bool]
    reasons: list[str]


def check_quality(
    candidate: dict[str, float],
    champion: dict[str, float],
) -> GateResult:
    """Check candidate quality against a champion."""
    checks = {
        "minimum_roc_auc": candidate["roc_auc"] >= MIN_ROC_AUC,
        "minimum_pr_auc": candidate["pr_auc"] >= MIN_PR_AUC,
        "minimum_recall": candidate["recall"] >= MIN_RECALL,
        "roc_auc_regression": (
            candidate["roc_auc"] >= champion["roc_auc"] - MAX_REGRESSION
        ),
        "pr_auc_regression": (
            candidate["pr_auc"] >= champion["pr_auc"] - MAX_REGRESSION
        ),
    }

    reasons = [name for name, passed in checks.items() if not passed]

    return GateResult(
        passed=all(checks.values()),
        checks=checks,
        reasons=reasons,
    )
