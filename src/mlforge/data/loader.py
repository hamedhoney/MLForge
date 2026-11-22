"""Load and validate the customer churn dataset."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

TARGET = "Churn"
ID_COLUMN = "customerID"


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the IBM Telco Churn dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

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

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=[TARGET, ID_COLUMN])

    return X, y
