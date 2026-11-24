import os
from pathlib import Path

import kagglehub
import pandas as pd

from mlforge.data.provenance import sha256_file

DATASET_HANDLE = "blastchar/telco-customer-churn"
DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"


def download_dataset() -> Path:
    """Download and verify the dataset using KaggleHub."""
    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))

    csv_path = dataset_dir / DATASET_FILENAME

    if not csv_path.is_file():
        matches = sorted(dataset_dir.rglob(DATASET_FILENAME))

        if not matches:
            raise FileNotFoundError(f"Expected {DATASET_FILENAME} in {dataset_dir}")

        csv_path = matches[0]

    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise ValueError(f"Dataset file is missing or empty: {csv_path}")

    # Verify dataset integrity when a checksum is configured.
    expected_hash = os.getenv("MLFORGE_DATASET_SHA256")

    if expected_hash:
        expected_hash = expected_hash.strip().lower()

        if len(expected_hash) != 64 or any(
            char not in "0123456789abcdef" for char in expected_hash
        ):
            raise ValueError(
                "MLFORGE_DATASET_SHA256 must be a "
                "64-character hexadecimal SHA-256 digest"
            )

        actual_hash = sha256_file(csv_path)

        if actual_hash.lower() != expected_hash:
            raise ValueError(
                "Dataset checksum mismatch. "
                f"Expected {expected_hash}, got {actual_hash}"
            )

        print(
            "Dataset SHA-256 verification passed.",
            flush=True,
        )

    print(
        f"Dataset downloaded: {csv_path}",
        flush=True,
    )

    return csv_path


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Download, validate, and prepare the churn dataset."""
    csv_path = download_dataset()

    df = pd.read_csv(csv_path)

    required = {
        TARGET,
        ID_COLUMN,
        "TotalCharges",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Dataset is empty")

    if df[ID_COLUMN].isna().any() or df[ID_COLUMN].duplicated().any():
        raise ValueError("Customer identifiers must be unique and non-null")

    if df[TARGET].isna().any() or not df[TARGET].isin(["Yes", "No"]).all():
        raise ValueError("Churn must contain only Yes/No values")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    y = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)

    X = df.drop(columns=[TARGET, ID_COLUMN])

    print(
        f"Dataset loaded: {len(X)} rows, {X.shape[1]} features",
        flush=True,
    )

    return X, y
