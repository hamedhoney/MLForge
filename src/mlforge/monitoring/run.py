import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from mlforge.data.loader import load_dataset
from mlforge.monitoring.drift import detect_drift

OUTPUT_DIR = Path("artifacts/monitoring")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, _ = load_dataset()

    reference, current = train_test_split(
        X,
        test_size=0.3,
        random_state=42,
    )

    current = current.copy()
    rng = np.random.default_rng(42)

    # Deliberately simulate changes in customer behavior.
    current["tenure"] = np.maximum(
        0,
        current["tenure"].to_numpy() - 15,
    )

    current["MonthlyCharges"] = current["MonthlyCharges"] * 1.25

    mask = rng.random(len(current)) < 0.5
    current.loc[mask, "Contract"] = "Month-to-month"

    report = detect_drift(reference, current)

    report["timestamp"] = datetime.now(UTC).isoformat()
    report["scenario"] = "simulated_distribution_shift"
    report["reference_samples"] = len(reference)
    report["current_samples"] = len(current)
    report["simulation"] = {
        "tenure": "Subtract 15 months, floor at zero",
        "MonthlyCharges": "Increase by 25%",
        "Contract": "Set 50% of sampled rows to month-to-month",
    }

    report_path = OUTPUT_DIR / "drift_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"Drifted features: {report['drifted_feature_count']}"
        f"/{report['total_features']}",
        flush=True,
    )
    print(f"Report saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
