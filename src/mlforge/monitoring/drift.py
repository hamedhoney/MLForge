import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    alpha: float = 0.05,
) -> dict:
    """Compare feature distributions using statistical tests."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    if reference.empty or current.empty:
        raise ValueError("Datasets must not be empty")

    if set(reference.columns) != set(current.columns):
        raise ValueError("Dataset schemas do not match")

    results = {}

    for column in reference.columns:
        ref = reference[column].dropna()
        cur = current[column].dropna()

        reference_missing_pct = float(reference[column].isna().mean() * 100)
        current_missing_pct = float(current[column].isna().mean() * 100)

        if ref.empty and cur.empty:
            p_value = 1.0
            statistic = 0.0
            test_name = "no_valid_data"

        elif ref.empty or cur.empty:
            p_value = 0.0
            statistic = None
            test_name = "insufficient_data"

        elif pd.api.types.is_numeric_dtype(reference[column]):
            test_name = "kolmogorov_smirnov"

            ref_values = ref.to_numpy(dtype=float)
            cur_values = cur.to_numpy(dtype=float)

            if np.array_equal(
                np.sort(ref_values),
                np.sort(cur_values),
            ):
                p_value = 1.0
                statistic = 0.0
            else:
                test = ks_2samp(ref_values, cur_values)
                p_value = float(test.pvalue)
                statistic = float(test.statistic)

        else:
            test_name = "chi_square"

            ref_counts = ref.astype(str).value_counts()
            cur_counts = cur.astype(str).value_counts()

            categories = ref_counts.index.union(cur_counts.index)

            counts = np.vstack(
                [
                    ref_counts.reindex(categories, fill_value=0).to_numpy(),
                    cur_counts.reindex(categories, fill_value=0).to_numpy(),
                ]
            )

            if counts.shape[1] < 2:
                p_value = 1.0
                statistic = 0.0
            else:
                test = chi2_contingency(counts)
                p_value = float(test.pvalue)
                statistic = float(test.statistic)

        if not np.isfinite(p_value):
            p_value = 0.0

        results[column] = {
            "test": test_name,
            "statistic": statistic,
            "p_value": p_value,
            "reference_missing_pct": reference_missing_pct,
            "current_missing_pct": current_missing_pct,
            "missing_pct_change": (current_missing_pct - reference_missing_pct),
        }

    # Benjamini-Hochberg false discovery rate correction.
    names = list(results)
    p_values = np.array([results[name]["p_value"] for name in names])

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    n_tests = len(sorted_p)

    adjusted_sorted = np.minimum.accumulate(
        (sorted_p * n_tests / np.arange(1, n_tests + 1))[::-1]
    )[::-1]

    adjusted = np.empty(n_tests)
    adjusted[order] = np.clip(adjusted_sorted, 0, 1)

    for i, name in enumerate(names):
        results[name]["adjusted_p_value"] = float(adjusted[i])
        results[name]["drift_detected"] = bool(adjusted[i] < alpha)

    drifted = [name for name, result in results.items() if result["drift_detected"]]

    return {
        "dataset_drift_detected": bool(drifted),
        "drifted_feature_count": len(drifted),
        "total_features": len(results),
        "drift_share": len(drifted) / len(results),
        "drifted_features": drifted,
        "features": results,
    }
