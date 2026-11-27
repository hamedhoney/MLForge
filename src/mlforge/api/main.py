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
from mlforge.monitoring.telemetry import InferenceTelemetry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and create a fresh collector for this process."""
    app.state.model_service = None
    app.state.model_error = None
    app.state.telemetry = InferenceTelemetry()
    try:
        app.state.model_service = ModelService()
    except (FileNotFoundError, ValueError, TypeError) as exc:
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="MLForge API",
    description="Customer churn prediction with process-local monitoring",
    version="0.4.0",
    lifespan=lifespan,
)


def get_service(request: Request) -> ModelService:
    """Return the loaded model or a service error."""
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
    """Time one inference call; record each returned prediction."""
    collector: InferenceTelemetry = request.app.state.telemetry
    try:
        service = get_service(request)
        start = perf_counter()
        results = service.predict(customers)
        latency_ms = (perf_counter() - start) * 1000
        if len(results) != len(customers):
            raise ValueError("Prediction count does not match request size")
        for result in results:
            collector.record(
                model_version=str(service.metadata.get("version", "unknown")),
                latency_ms=latency_ms / len(results),
                probability=result.churn_probability,
                prediction=result.prediction,
            )
        return results
    except Exception:
        collector.record_error()
        raise


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
    """Return aggregate metrics only when monitoring access is configured."""
    expected = os.getenv("MLFORGE_MONITORING_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Monitoring access not configured")
    if not x_monitoring_token or not secrets.compare_digest(
        x_monitoring_token, expected
    ):
        raise HTTPException(status_code=403, detail="Forbidden")
    return request.app.state.telemetry.summary()
