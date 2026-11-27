import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlforge.api.schemas import CustomerFeatures, PredictionResponse
from mlforge.deployment import download_model

logger = logging.getLogger(__name__)

MODEL_DIR = Path(
    os.getenv("MLFORGE_MODEL_DIR", "artifacts/tuning")
)


class ModelService:
    """Inference wrapper around a verified MLForge model."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        model_dir = Path(model_dir)

        model_path = model_dir / "best_model.joblib"
        metadata_path = model_dir / "metadata.json"

        # Download the model when artifacts are missing.
        if not model_path.is_file() or not metadata_path.is_file():
            logger.info(
                "Model artifacts missing. Downloading verified release."
            )

            # The downloader uses MLFORGE_MODEL_DIR.
            # Ensure a custom directory stays consistent.
            if model_dir != MODEL_DIR:
                raise ValueError(
                    "Custom model directory requires preloaded artifacts"
                )

            version = download_model()

            logger.info(
                "Downloaded MLForge model version %s",
                version,
            )

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model artifact missing: {model_path}"
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Model metadata missing: {metadata_path}"
            )

        logger.info("Loading model from %s", model_path)

        # Only load trusted, verified model artifacts.
        # joblib deserialization can execute arbitrary code.
        self.model = joblib.load(model_path)

        self.metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        self.threshold = float(self.metadata["threshold"])
        self.model_name = self.metadata["model"]
        self.features = self.metadata["features"]

        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid model threshold")

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "Loaded model does not support predict_proba"
            )

        version_path = model_dir / "version.txt"

        self.model_version = (
            version_path.read_text(encoding="utf-8").strip()
            if version_path.is_file()
            else os.getenv("MLFORGE_MODEL_VERSION", "unknown")
        )

        logger.info(
            "MLForge model loaded successfully: %s (%s)",
            self.model_name,
            self.model_version,
        )

    def predict(
        self,
        customers: list[CustomerFeatures],
    ) -> list[PredictionResponse]:
        """Predict churn for one or more customers."""
        if not customers:
            return []

        records = [
            customer.model_dump()
            for customer in customers
        ]

        frame = pd.DataFrame(records)
        frame = frame.reindex(columns=self.features)

        probabilities = np.asarray(
            self.model.predict_proba(frame)[:, 1]
        )

        results = []

        for probability in probabilities:
            probability = float(probability)

            if not np.isfinite(probability):
                raise ValueError("Model returned a non-finite probability")

            if not 0 <= probability <= 1:
                raise ValueError("Model returned an invalid probability")

            prediction = int(
                probability >= self.threshold
            )

            if probability >= 0.7:
                risk = "high"
            elif probability >= 0.3:
                risk = "medium"
            else:
                risk = "low"

            results.append(
                PredictionResponse(
                    prediction=prediction,
                    churn_probability=probability,
                    threshold=self.threshold,
                    risk_level=risk,
                    model_name=self.model_name,
                )
            )

        return results
