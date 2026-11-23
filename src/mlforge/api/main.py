from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from mlforge.api.schemas import (
    BatchPredictionRequest,
    CustomerFeatures,
    PredictionResponse,
)
from mlforge.api.service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the service starts."""
    app.state.model_service = None
    app.state.model_error = None

    try:
        app.state.model_service = ModelService()
    except (FileNotFoundError, ValueError, TypeError) as exc:
        app.state.model_error = str(exc)

    yield


app = FastAPI(
    title="MLForge API",
    description="Production-style customer churn prediction",
    version="0.1.0",
    lifespan=lifespan,
)


def get_service(request: Request) -> ModelService:
    """Return the loaded model or a service error."""
    service = request.app.state.model_service

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        )

    return service


@app.get("/health")
def health(request: Request):
    ready = request.app.state.model_service is not None

    return {
        "status": "ready" if ready else "unavailable",
        "model_loaded": ready,
    }


@app.get("/model/info")
def model_info(request: Request):
    service = get_service(request)

    return {
        "model_name": service.model_name,
        "threshold": service.threshold,
        "features": service.features,
        "feature_count": len(service.features),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request):
    service = get_service(request)
    return service.predict([customer])[0]


@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
):
    service = get_service(request)
    return service.predict(payload.customers)
