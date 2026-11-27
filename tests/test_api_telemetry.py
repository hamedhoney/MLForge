"""Integration tests for API inference telemetry without a real model artifact."""

import pytest
from fastapi.testclient import TestClient

from mlforge.api.main import app
from mlforge.api.schemas import PredictionResponse

CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
}


class FakeService:
    model_name = "fake"
    threshold = 0.5
    features = list(CUSTOMER)
    metadata = {"version": "v0.3.0"}

    def predict(self, customers):
        return [
            PredictionResponse(
                prediction=1,
                churn_probability=0.8,
                threshold=0.5,
                risk_level="high",
                model_name="fake",
            )
            for _ in customers
        ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("mlforge.api.main.ModelService", FakeService)
    monkeypatch.setenv("MLFORGE_MONITORING_TOKEN", "test-secret")
    with TestClient(app) as test_client:
        yield test_client


def summary(client):
    response = client.get(
        "/monitoring/summary", headers={"X-Monitoring-Token": "test-secret"}
    )
    assert response.status_code == 200
    return response.json()


def test_single_prediction_is_recorded(client):
    response = client.post("/predict", json=CUSTOMER)
    assert response.status_code == 200
    assert response.json()["churn_probability"] == 0.8
    assert summary(client)["total_predictions"] == 1
    assert summary(client)["model_versions"] == ["v0.3.0"]


def test_batch_counts_predictions(client):
    response = client.post("/predict/batch", json={"customers": [CUSTOMER] * 3})
    assert response.status_code == 200
    assert len(response.json()) == 3
    assert summary(client)["total_predictions"] == 3


def test_invalid_request_is_not_recorded(client):
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422
    assert summary(client)["total_predictions"] == 0


def test_monitoring_endpoint_requires_token(client):
    assert client.get("/monitoring/summary").status_code == 403
    assert (
        client.get(
            "/monitoring/summary", headers={"X-Monitoring-Token": "wrong"}
        ).status_code
        == 403
    )
