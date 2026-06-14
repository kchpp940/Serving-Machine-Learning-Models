import sys
import os

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
FASTAPI_DIR = os.path.join(PROJECT_ROOT, "fastapi")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FASTAPI_DIR)

import importlib.util
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

spec = importlib.util.spec_from_file_location("app_module", os.path.join(FASTAPI_DIR, "app.py"))
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
app = app_module.app

spec_models = importlib.util.spec_from_file_location("models_module", os.path.join(FASTAPI_DIR, "models.py"))
models_module = importlib.util.module_from_spec(spec_models)
spec_models.loader.exec_module(models_module)
PredictionResponse = models_module.PredictionResponse
ErrorResponse = models_module.ErrorResponse
HealthResponse = models_module.HealthResponse


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_endpoint_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "ok"
        assert "service" in data
        assert "version" in data
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_response_matches_schema(self):
        response = client.get("/health")
        data = response.json()
        try:
            HealthResponse(**data)
        except ValidationError as e:
            pytest.fail(f"Health response does not match schema: {e}")


class TestSchemaEndpoint:
    def test_schema_endpoint_returns_data(self):
        response = client.get("/schema")
        assert response.status_code == 200
        data = response.json()

        required_fields = [
            "feature_order",
            "numeric_features",
            "categorical_features",
            "categorical_options",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        assert isinstance(data["feature_order"], list)
        assert len(data["feature_order"]) > 0


class TestPredictEndpoint:
    def test_predict_success_response_format(self):
        test_data = {
            "enginesize": 130,
            "curbweight": 2548,
            "horsepower": 111,
            "highwaympg": 27,
            "carwidth": 64.1,
            "wheelbase": 88.6,
            "drivewheel": "rwd",
            "citympg": 21,
            "boreratio": 3.47,
            "cylindernumber": "four",
        }

        response = client.post("/predict", json=test_data)
        assert response.status_code == 200

        data = response.json()

        assert "prediction" in data
        assert isinstance(data["prediction"], float)
        assert data["prediction"] > 0
        assert "currency" in data
        assert isinstance(data["currency"], str)
        assert "model_name" in data
        assert isinstance(data["model_name"], str)

    def test_predict_response_matches_schema(self):
        test_data = {
            "enginesize": 130,
            "curbweight": 2548,
            "horsepower": 111,
            "highwaympg": 27,
            "carwidth": 64.1,
            "wheelbase": 88.6,
            "drivewheel": "rwd",
            "citympg": 21,
            "boreratio": 3.47,
            "cylindernumber": "four",
        }

        response = client.post("/predict", json=test_data)
        data = response.json()
        try:
            PredictionResponse(**data)
        except ValidationError as e:
            pytest.fail(f"Prediction response does not match schema: {e}")

    def test_predict_response_has_no_array_index(self):
        test_data = {
            "enginesize": 130,
            "curbweight": 2548,
            "horsepower": 111,
            "highwaympg": 27,
            "carwidth": 64.1,
            "wheelbase": 88.6,
            "drivewheel": "rwd",
            "citympg": 21,
            "boreratio": 3.47,
            "cylindernumber": "four",
        }

        response = client.post("/predict", json=test_data)
        data = response.json()

        assert not isinstance(data, list), "Response should be a JSON object, not a list"
        assert "prediction" in data, "Response should have 'prediction' field"

    def test_predict_invalid_input_returns_error_format(self):
        test_data = {
            "enginesize": "not_a_number",
            "curbweight": 2548,
            "horsepower": 111,
            "highwaympg": 27,
            "carwidth": 64.1,
            "wheelbase": 88.6,
            "drivewheel": "rwd",
            "citympg": 21,
            "boreratio": 3.47,
            "cylindernumber": "four",
        }

        response = client.post("/predict", json=test_data)
        assert response.status_code == 400

        data = response.json()
        assert "status" in data
        assert data["status"] == "error"
        assert "error" in data
        assert "detail" in data

    def test_predict_missing_field_returns_error_format(self):
        test_data = {
            "enginesize": 130,
        }

        response = client.post("/predict", json=test_data)
        assert response.status_code == 400

        data = response.json()
        assert "status" in data
        assert data["status"] == "error"
        assert "error" in data
        assert "detail" in data

    def test_predict_invalid_categorical_value(self):
        test_data = {
            "enginesize": 130,
            "curbweight": 2548,
            "horsepower": 111,
            "highwaympg": 27,
            "carwidth": 64.1,
            "wheelbase": 88.6,
            "drivewheel": "invalid_value",
            "citympg": 21,
            "boreratio": 3.47,
            "cylindernumber": "four",
        }

        response = client.post("/predict", json=test_data)
        assert response.status_code == 400

        data = response.json()
        assert "status" in data
        assert data["status"] == "error"
        assert "detail" in data


class TestResponseProtocolConsistency:
    def test_predict_success_has_model_semantic_fields_only(self):
        test_data = {
            "enginesize": 130,
            "curbweight": 2548,
            "horsepower": 111,
            "highwaympg": 27,
            "carwidth": 64.1,
            "wheelbase": 88.6,
            "drivewheel": "rwd",
            "citympg": 21,
            "boreratio": 3.47,
            "cylindernumber": "four",
        }
        predict_resp = client.post("/predict", json=test_data)
        predict_data = predict_resp.json()

        assert "prediction" in predict_data
        assert "currency" in predict_data
        assert "model_name" in predict_data
        assert "status" not in predict_data, "Success response must not carry transport-level status field"
        assert "message" not in predict_data, "Success response must not carry message field"

    def test_error_response_has_consistent_status_envelope(self):
        resp = client.post("/predict", json={"enginesize": 130})
        err_data = resp.json()
        assert err_data["status"] == "error"
        assert "error" in err_data
        assert "detail" in err_data

    def test_health_response_has_consistent_status(self):
        health_resp = client.get("/health")
        health_data = health_resp.json()
        assert health_data["status"] == "ok"
        assert "service" in health_data
        assert "version" in health_data
        assert "model_loaded" in health_data

    def test_prediction_field_is_numeric(self):
        test_data = {
            "enginesize": 130,
            "curbweight": 2548,
            "horsepower": 111,
            "highwaympg": 27,
            "carwidth": 64.1,
            "wheelbase": 88.6,
            "drivewheel": "rwd",
            "citympg": 21,
            "boreratio": 3.47,
            "cylindernumber": "four",
        }

        response = client.post("/predict", json=test_data)
        data = response.json()

        prediction = data["prediction"]
        assert isinstance(prediction, (int, float)), "prediction must be a number"
