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

MODEL_DIR = Path(os.getenv("MLFORGE_MODEL_DIR", "artifacts/tuning"))


class ModelService:
    """Inference wrapper for a verified, versioned model."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        model_dir = Path(model_dir)

        model_path = model_dir / "best_model.joblib"
        metadata_path = model_dir / "metadata.json"
        version_path = model_dir / "version.txt"

        requested_version = os.getenv("MLFORGE_MODEL_VERSION")

        cached_version = (
            version_path.read_text(encoding="utf-8").strip()
            if version_path.is_file()
            else None
        )

        missing = (
            not model_path.is_file()
            or not metadata_path.is_file()
            or not version_path.is_file()
        )

        version_mismatch = (
            requested_version is not None and cached_version != requested_version
        )

        if missing or version_mismatch:
            if model_dir != MODEL_DIR:
                raise ValueError("Custom model directory requires preloaded artifacts")

            logger.info(
                "Downloading model: requested=%s cached=%s",
                requested_version,
                cached_version,
            )
            download_model()

        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        if not version_path.is_file():
            raise FileNotFoundError(version_path)

        self.model_version = version_path.read_text(encoding="utf-8").strip()

        if requested_version and self.model_version != requested_version:
            raise RuntimeError("Loaded model version mismatch")

        # Only deserialize trusted, verified model artifacts.
        self.model = joblib.load(model_path)

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.threshold = float(self.metadata["threshold"])
        self.model_name = self.metadata["model"]
        self.features = self.metadata["features"]

        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid model threshold")

        if (
            not isinstance(self.features, list)
            or not self.features
            or len(self.features) != len(set(self.features))
        ):
            raise ValueError("Invalid feature schema")

        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Loaded model does not support predict_proba")

        logger.info(
            "Loaded MLForge model %s (%s)",
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

        records = [customer.model_dump() for customer in customers]

        frame = pd.DataFrame(records).reindex(columns=self.features)

        probabilities = np.asarray(self.model.predict_proba(frame)[:, 1])

        results = []

        for value in probabilities:
            probability = float(value)

            if not np.isfinite(probability):
                raise ValueError("Non-finite model probability")

            if not 0 <= probability <= 1:
                raise ValueError("Invalid model probability")

            prediction = int(probability >= self.threshold)

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
