import logging
import os
import secrets
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Header, HTTPException, Request
from sqlalchemy import text

from mlforge.api.schemas import (
    BatchPredictionRequest,
    CustomerFeatures,
    PredictionResponse,
)
from mlforge.api.service import ModelService
from mlforge.monitoring.storage import TelemetryStore
from mlforge.monitoring.telemetry import InferenceTelemetry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connect to an already-migrated telemetry database."""
    app.state.model_service = None
    app.state.model_error = None
    app.state.telemetry = InferenceTelemetry()
    app.state.telemetry_store = None

    try:
        app.state.model_service = ModelService()
    except Exception:
        logger.exception("Model initialization failed")
        app.state.model_error = "Model initialization failed"

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        store = None
        try:
            if database_url.startswith("postgresql://"):
                database_url = database_url.replace(
                    "postgresql://", "postgresql+psycopg://", 1
                )
            store = TelemetryStore(database_url)
            with store.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            # No create_all here: run the initial schema migration separately.
            app.state.telemetry_store = store
            logger.info("Persistent telemetry database connected")
        except Exception:
            logger.exception("Persistent telemetry database unavailable")
            if store is not None:
                store.close()
    try:
        yield
    finally:
        if app.state.telemetry_store is not None:
            app.state.telemetry_store.close()


app = FastAPI(
    title="MLForge API",
    description="Customer churn prediction with persistent monitoring",
    version="0.4.0",
    lifespan=lifespan,
)


def get_service(request: Request) -> ModelService:
    service = request.app.state.model_service
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    return service


@app.get("/health")
def health(request: Request):
    ready = request.app.state.model_service is not None
    return {"status": "ready" if ready else "unavailable", "model_loaded": ready}


@app.get("/model/info")
def model_info(request: Request):
    service = get_service(request)
    return {
        "model_name": service.model_name,
        "threshold": service.threshold,
        "features": service.features,
        "feature_count": len(service.features),
    }


def _predict_and_record(customers: list[CustomerFeatures], request: Request):
    collector: InferenceTelemetry = request.app.state.telemetry
    try:
        service = get_service(request)
        start = perf_counter()
        results = service.predict(customers)
        latency_ms = (perf_counter() - start) * 1000
        if len(results) != len(customers):
            raise ValueError("Prediction count does not match request size")
    except Exception:
        collector.record_error()
        raise

    version = str(getattr(service, "model_version", "unknown"))
    per_prediction_ms = latency_ms / len(results) if results else 0.0
    store = getattr(request.app.state, "telemetry_store", None)
    for result in results:
        event = {
            "model_version": version,
            "latency_ms": per_prediction_ms,
            "probability": result.churn_probability,
            "prediction": result.prediction,
        }
        collector.record(**event)
        if store is not None:
            try:
                store.record(**event)
            except Exception:
                logger.exception("Failed to persist inference telemetry")
    return results


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request):
    return _predict_and_record([customer], request)[0]


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(payload: BatchPredictionRequest, request: Request):
    return _predict_and_record(payload.customers, request)


@app.get("/monitoring/summary")
def monitoring_summary(
    request: Request, x_monitoring_token: str | None = Header(default=None)
):
    expected = os.getenv("MLFORGE_MONITORING_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Monitoring access not configured")
    if not x_monitoring_token or not secrets.compare_digest(
        x_monitoring_token, expected
    ):
        raise HTTPException(status_code=403, detail="Forbidden")

    store = getattr(request.app.state, "telemetry_store", None)
    if store is not None:
        try:
            historical = store.summary()
            return {
                **historical,
                "source": "persistent_database",
                "process_error_count": request.app.state.telemetry.summary()[
                    "error_count"
                ],
            }
        except Exception:
            logger.exception("Failed to read persistent telemetry")
            raise HTTPException(
                status_code=503, detail="Monitoring storage unavailable"
            ) from None
    return {**request.app.state.telemetry.summary(), "source": "process_memory"}
