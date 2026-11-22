from pathlib import Path

import kagglehub
import pandas as pd

DATASET_HANDLE = "blastchar/telco-customer-churn"
DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"


def download_dataset() -> Path:
    """Download the dataset to the cloud runner's KaggleHub cache."""
    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))

    csv_path = dataset_dir / DATASET_FILENAME

    if not csv_path.is_file():
        matches = list(dataset_dir.rglob(DATASET_FILENAME))

        if not matches:
            raise FileNotFoundError(f"Expected {DATASET_FILENAME} in {dataset_dir}")

        csv_path = matches[0]

    print(f"Dataset downloaded: {csv_path}", flush=True)

    return csv_path


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Download, validate, and prepare the churn dataset."""
    csv_path = download_dataset()

    df = pd.read_csv(csv_path)

    required = {TARGET, ID_COLUMN, "TotalCharges"}
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
