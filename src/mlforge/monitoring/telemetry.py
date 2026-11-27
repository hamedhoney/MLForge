from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import time

import numpy as np


@dataclass(frozen=True)
class InferenceEvent:
    timestamp: float
    model_version: str
    latency_ms: float
    probability: float
    prediction: int


class InferenceTelemetry:
    """Collect bounded inference telemetry without customer records."""

    def __init__(self, max_events: int = 10000):
        if max_events < 1:
            raise ValueError("max_events must be positive")

        self._events: deque[InferenceEvent] = deque(maxlen=max_events)
        self._lock = Lock()
        self._request_count = 0
        self._error_count = 0

    def record(
        self,
        *,
        model_version: str,
        latency_ms: float,
        probability: float,
        prediction: int,
    ) -> None:
        """Record a successful individual prediction."""
        if not np.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("Invalid latency")

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid probability")

        if prediction not in (0, 1):
            raise ValueError("Prediction must be 0 or 1")

        event = InferenceEvent(
            timestamp=time(),
            model_version=model_version,
            latency_ms=float(latency_ms),
            probability=float(probability),
            prediction=int(prediction),
        )

        with self._lock:
            self._events.append(event)
            self._request_count += 1

    def record_error(self) -> None:
        """Record a failed inference request."""
        with self._lock:
            self._error_count += 1

    def summary(self) -> dict:
        """Return aggregate statistics."""
        with self._lock:
            events = list(self._events)
            total = self._request_count
            errors = self._error_count

        if not events:
            return {
                "total_predictions": total,
                "error_count": errors,
                "retained_events": 0,
                "mean_latency_ms": None,
                "p50_latency_ms": None,
                "p95_latency_ms": None,
                "mean_probability": None,
                "predicted_churn_rate": None,
            }

        latencies = np.array([event.latency_ms for event in events])
        probabilities = np.array([event.probability for event in events])
        predictions = np.array([event.prediction for event in events])

        return {
            "total_predictions": total,
            "error_count": errors,
            "retained_events": len(events),
            "mean_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "mean_probability": float(np.mean(probabilities)),
            "predicted_churn_rate": float(np.mean(predictions)),
            "model_versions": sorted({event.model_version for event in events}),
        }


telemetry = InferenceTelemetry()
