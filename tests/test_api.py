from unittest.mock import patch

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
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.5,
    "TotalCharges": 950.0,
}


def test_health_without_model():
    with (
        patch(
            "mlforge.api.main.ModelService",
            side_effect=FileNotFoundError("Missing model"),
        ),
        TestClient(app) as client,
    ):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is False

        unavailable = client.post("/predict", json=CUSTOMER)
        assert unavailable.status_code == 503


def test_prediction_with_mock_model():
    with patch("mlforge.api.main.ModelService") as service:
        service.return_value.predict.return_value = [
            PredictionResponse(
                prediction=1,
                churn_probability=0.78,
                threshold=0.33,
                risk_level="high",
                model_name="xgboost",
            )
        ]

        with TestClient(app) as client:
            response = client.post("/predict", json=CUSTOMER)

            assert response.status_code == 200
            assert response.json()["prediction"] == 1
            assert response.json()["churn_probability"] == 0.78


def test_invalid_customer_is_rejected():
    with patch("mlforge.api.main.ModelService"), TestClient(app) as client:
        response = client.post(
            "/predict",
            json={**CUSTOMER, "tenure": -5},
        )

        assert response.status_code == 422
