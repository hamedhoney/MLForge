import logging
import os
import secrets
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Header, HTTPException, Request

from mlforge.api.schemas import (
    BatchPredictionRequest,
    CustomerFeatures,
    PredictionResponse,
)
from mlforge.api.service import ModelService
from mlforge.monitoring.storage import TelemetryStore
from mlforge.monitoring.telemetry import InferenceTelemetry

logger = logging.getLogger(__name__)


def get_database_url() -> str | None:
    """Return the configured SQLAlchemy database URL."""
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        return None

    if database_url.startswith("postgresql://"):
        return database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    return database_url


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model and optional persistent telemetry."""
    app.state.model_service = None
    app.state.model_error = None
    app.state.telemetry = InferenceTelemetry()
    app.state.telemetry_store = None

    try:
        app.state.model_service = ModelService()
        logger.info("MLForge model initialized successfully")
    except Exception:
        logger.exception("MLForge model initialization failed")
        app.state.model_error = "Model initialization failed"

    database_url = get_database_url()

    if database_url:
        try:
            store = TelemetryStore(database_url)

            # Schema initialization is performed separately
            # through the init-database GitHub Actions workflow.
            app.state.telemetry_store = store

            logger.info("Persistent telemetry storage configured")
        except Exception:
            logger.exception("Failed to configure telemetry storage")
    else:
        logger.warning("DATABASE_URL not configured; using in-memory telemetry")

    try:
        yield
    finally:
        store = app.state.telemetry_store
        if store is not None:
            store.close()


app = FastAPI(
    title="MLForge API",
    description="Customer churn prediction with persistent monitoring",
    version="0.4.0",
    lifespan=lifespan,
)


def get_service(request: Request) -> ModelService:
    """Return the loaded model or raise a service error."""
    service = request.app.state.model_service

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        )

    return service


def get_model_version(service: ModelService) -> str:
    """Resolve the model version from service or metadata."""
    version = getattr(service, "model_version", None)

    if not version:
        metadata = getattr(service, "metadata", {})
        version = metadata.get("version", "unknown")

    return str(version)


def record_persistent_prediction(
    request: Request,
    *,
    model_version: str,
    latency_ms: float,
    probability: float,
    prediction: int,
) -> None:
    """Persist telemetry without interrupting model inference."""
    store = getattr(request.app.state, "telemetry_store", None)

    if store is None:
        return

    try:
        store.record(
            model_version=model_version,
            latency_ms=latency_ms,
            probability=probability,
            prediction=prediction,
        )
    except Exception:
        logger.exception("Failed to persist inference telemetry")


def _predict_and_record(
    customers: list[CustomerFeatures],
    request: Request,
) -> list[PredictionResponse]:
    """Execute inference and record individual prediction events."""
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

    model_version = get_model_version(service)

    if not results:
        return results

    # Allocate batch execution time across returned predictions.
    per_prediction_latency = latency_ms / len(results)

    for result in results:
        collector.record(
            model_version=model_version,
            latency_ms=per_prediction_latency,
            probability=result.churn_probability,
            prediction=result.prediction,
        )

        record_persistent_prediction(
            request,
            model_version=model_version,
            latency_ms=per_prediction_latency,
            probability=result.churn_probability,
            prediction=result.prediction,
        )

    return results


@app.get("/health")
def health(request: Request):
    """Return model readiness."""
    ready = request.app.state.model_service is not None

    return {
        "status": "ready" if ready else "unavailable",
        "model_loaded": ready,
    }


@app.get("/model/info")
def model_info(request: Request):
    """Return model information."""
    service = get_service(request)

    return {
        "model_name": service.model_name,
        "threshold": service.threshold,
        "features": service.features,
        "feature_count": len(service.features),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    customer: CustomerFeatures,
    request: Request,
):
    """Predict churn for one customer."""
    return _predict_and_record([customer], request)[0]


@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
):
    """Predict churn for multiple customers."""
    return _predict_and_record(payload.customers, request)


@app.get("/monitoring/summary")
def monitoring_summary(
    request: Request,
    x_monitoring_token: str | None = Header(default=None),
    days: int = 30,
):
    """Return authenticated historical inference statistics."""
    expected = os.getenv("MLFORGE_MONITORING_TOKEN")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Monitoring access not configured",
        )

    if not x_monitoring_token or not secrets.compare_digest(
        x_monitoring_token,
        expected,
    ):
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    if not 1 <= days <= 365:
        raise HTTPException(
            status_code=422,
            detail="days must be between 1 and 365",
        )

    collector: InferenceTelemetry = request.app.state.telemetry
    store = getattr(request.app.state, "telemetry_store", None)

    if store is not None:
        try:
            summary = store.summary(days=days)

            return {
                "source": "persistent_database",
                **summary,
                "process_error_count": collector.summary()["error_count"],
            }

        except Exception:
            logger.exception("Failed to retrieve persistent telemetry summary")

    return {
        "source": "process_memory",
        **collector.summary(),
    }
