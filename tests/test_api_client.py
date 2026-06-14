import sys
import os
import json

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import Mock, patch
import requests

from car_pricing.api_client import (
    PredictionAPIClient,
    PredictionResult,
    HealthResult,
    SchemaResult,
    APIClientError,
    APIConnectionError,
    APITimeoutError,
    APIHTTPError,
    APIInvalidResponseError,
    ENV_API_BASE_URL,
    ENV_API_TIMEOUT,
    DEFAULT_API_BASE_URL,
    DEFAULT_TIMEOUT,
    HEALTH_PATH,
    PREDICT_PATH,
    SCHEMA_PATH,
)


TEST_FEATURES = {
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


class TestClientConfiguration:
    def test_default_config_from_env(self, monkeypatch):
        monkeypatch.setenv(ENV_API_BASE_URL, "https://test.api.example.com")
        monkeypatch.setenv(ENV_API_TIMEOUT, "30")

        client = PredictionAPIClient()
        assert client.base_url == "https://test.api.example.com"
        assert client.timeout == 30

    def test_default_config_when_no_env(self, monkeypatch):
        monkeypatch.delenv(ENV_API_BASE_URL, raising=False)
        monkeypatch.delenv(ENV_API_TIMEOUT, raising=False)

        client = PredictionAPIClient()
        assert client.base_url == DEFAULT_API_BASE_URL
        assert client.timeout == DEFAULT_TIMEOUT

    def test_explicit_config_overrides_env(self, monkeypatch):
        monkeypatch.setenv(ENV_API_BASE_URL, "https://ignored.example.com")
        monkeypatch.setenv(ENV_API_TIMEOUT, "999")

        client = PredictionAPIClient(base_url="http://explicit:8000", timeout=15)
        assert client.base_url == "http://explicit:8000"
        assert client.timeout == 15

    def test_base_url_trailing_slash_stripped(self):
        client = PredictionAPIClient(base_url="http://test.com/")
        assert client.base_url == "http://test.com"

    def test_paths_are_constants(self):
        assert HEALTH_PATH == "/health"
        assert PREDICT_PATH == "/predict"
        assert SCHEMA_PATH == "/schema"


class TestClientErrorHandling:
    @patch("requests.request")
    def test_connection_error_raises_api_connection_error(self, mock_request):
        mock_request.side_effect = requests.exceptions.ConnectionError("DNS failure")

        client = PredictionAPIClient(base_url="http://down.example.com")
        with pytest.raises(APIConnectionError) as exc_info:
            client.health_check()
        assert "Cannot connect" in str(exc_info.value)

    @patch("requests.request")
    def test_timeout_error_raises_api_timeout_error(self, mock_request):
        mock_request.side_effect = requests.exceptions.Timeout("Too slow")

        client = PredictionAPIClient(base_url="http://slow.example.com", timeout=1)
        with pytest.raises(APITimeoutError) as exc_info:
            client.health_check()
        assert "timed out" in str(exc_info.value)
        assert "1s" in str(exc_info.value)

    @patch("requests.request")
    def test_http_400_parses_error_and_detail(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "status": "error",
            "error": "ValidationError",
            "detail": "enginesize must be positive",
        }
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        with pytest.raises(APIHTTPError) as exc_info:
            client.predict(TEST_FEATURES)

        assert exc_info.value.status_code == 400
        assert exc_info.value.error_type == "ValidationError"
        assert exc_info.value.detail == "enginesize must be positive"

    @patch("requests.request")
    def test_http_404_parses_error(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "status": "error",
            "error": "NotFound",
            "detail": "Endpoint not found",
        }
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        with pytest.raises(APIHTTPError) as exc_info:
            client.predict(TEST_FEATURES)

        assert exc_info.value.status_code == 404
        assert exc_info.value.error_type == "NotFound"

    @patch("requests.request")
    def test_http_500_parses_error(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "status": "error",
            "error": "InternalServerError",
            "detail": "Database connection failed",
        }
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        with pytest.raises(APIHTTPError) as exc_info:
            client.predict(TEST_FEATURES)

        assert exc_info.value.status_code == 500
        assert exc_info.value.error_type == "InternalServerError"
        assert exc_info.value.detail == "Database connection failed"

    @patch("requests.request")
    def test_http_error_without_json_body(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        with pytest.raises(APIHTTPError) as exc_info:
            client.predict(TEST_FEATURES)

        assert exc_info.value.status_code == 400
        assert exc_info.value.error_type == "HTTP400"
        assert "HTTP Error 400" in exc_info.value.detail

    @patch("requests.request")
    def test_non_json_response_raises_invalid_response_error(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Not JSON")
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        with pytest.raises(APIInvalidResponseError) as exc_info:
            client.predict(TEST_FEATURES)
        assert "invalid JSON" in str(exc_info.value)

    @patch("requests.request")
    def test_list_response_raises_invalid_response_error(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [14740.21, "USD"]
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        with pytest.raises(APIInvalidResponseError) as exc_info:
            client.predict(TEST_FEATURES)
        assert "not a JSON object" in str(exc_info.value)

    @patch("requests.request")
    def test_missing_prediction_field_raises_error(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "currency": "USD",
            "model_name": "test_model",
        }
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        with pytest.raises(APIInvalidResponseError) as exc_info:
            client.predict(TEST_FEATURES)
        assert "missing 'prediction' field" in str(exc_info.value)

    @patch("requests.request")
    def test_non_numeric_prediction_raises_error(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": "not_a_number"}
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        with pytest.raises(APIInvalidResponseError) as exc_info:
            client.predict(TEST_FEATURES)
        assert "not a valid number" in str(exc_info.value)
        assert "not_a_number" in str(exc_info.value)

    @patch("requests.request")
    def test_no_array_index_access(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prediction": 14740.21,
            "currency": "USD",
            "model_name": "sklearn_gbr",
        }
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        result = client.predict(TEST_FEATURES)

        assert isinstance(result, PredictionResult)
        assert result.prediction == 14740.21
        assert result.currency == "USD"
        assert result.model_name == "sklearn_gbr"

        data = mock_response.json.return_value
        assert isinstance(data, dict), "Response must be dict, client must not use array indexing"
        assert "prediction" in data, "Client accesses prediction via dict key"
        assert isinstance(data["prediction"], float), "Client reads prediction as float from dict"

    def test_format_error_connection(self):
        error = APIConnectionError("Cannot connect")
        formatted = PredictionAPIClient.format_error(error)
        assert "❌ Connection Error" in formatted
        assert "Cannot connect" in formatted

    def test_format_error_timeout(self):
        error = APITimeoutError("Timed out after 10s")
        formatted = PredictionAPIClient.format_error(error)
        assert "⏱️ Timeout" in formatted

    def test_format_error_http_400(self):
        error = APIHTTPError(400, "ValidationError", "Invalid enginesize")
        formatted = PredictionAPIClient.format_error(error)
        assert "⚠️ Invalid Input" in formatted
        assert "Invalid enginesize" in formatted

    def test_format_error_http_404(self):
        error = APIHTTPError(404, "NotFound", "No such endpoint")
        formatted = PredictionAPIClient.format_error(error)
        assert "🔍 Not Found" in formatted

    def test_format_error_http_500(self):
        error = APIHTTPError(500, "InternalError", "Server crash")
        formatted = PredictionAPIClient.format_error(error)
        assert "💥 Server Error" in formatted

    def test_format_error_invalid_response(self):
        error = APIInvalidResponseError("Bad JSON")
        formatted = PredictionAPIClient.format_error(error)
        assert "📝 Invalid Response" in formatted

    def test_format_error_unexpected(self):
        error = RuntimeError("Something weird")
        formatted = PredictionAPIClient.format_error(error)
        assert "❌ Unexpected Error" in formatted
        assert "Something weird" in formatted


class TestClientSuccessScenarios:
    @patch("requests.request")
    def test_predict_success(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prediction": 14740.21,
            "currency": "USD",
            "model_name": "sklearn_gbr",
        }
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        result = client.predict(TEST_FEATURES)

        assert isinstance(result, PredictionResult)
        assert result.prediction == 14740.21
        assert result.currency == "USD"
        assert result.model_name == "sklearn_gbr"

        args, kwargs = mock_request.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["url"] == f"http://test.com{PREDICT_PATH}"
        assert kwargs["json"] == TEST_FEATURES
        assert kwargs["timeout"] == DEFAULT_TIMEOUT

    @patch("requests.request")
    def test_predict_default_currency_when_missing(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": 14740.21}
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        result = client.predict(TEST_FEATURES)

        assert result.prediction == 14740.21
        assert result.currency == "USD"
        assert result.model_name is None

    @patch("requests.request")
    def test_predict_integer_prediction_converted_to_float(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": 15000}
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        result = client.predict(TEST_FEATURES)

        assert isinstance(result.prediction, float)
        assert result.prediction == 15000.0

    @patch("requests.request")
    def test_health_check_success(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "service": "car-price-prediction-api",
            "version": "1.0.0",
            "model_loaded": True,
        }
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        result = client.health_check()

        assert isinstance(result, HealthResult)
        assert result.status == "ok"
        assert result.service == "car-price-prediction-api"
        assert result.version == "1.0.0"
        assert result.model_loaded is True

        args, kwargs = mock_request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == f"http://test.com{HEALTH_PATH}"

    @patch("requests.request")
    def test_get_schema_success(self, mock_request):
        schema_data = {
            "feature_order": ["enginesize", "horsepower"],
            "numeric_features": ["enginesize", "horsepower"],
            "categorical_features": ["drivewheel"],
            "target_column": "price",
            "categorical_options": {"drivewheel": ["rwd", "fwd"]},
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = schema_data
        mock_request.return_value = mock_response

        client = PredictionAPIClient(base_url="http://test.com")
        result = client.get_schema()

        assert isinstance(result, SchemaResult)
        assert result.feature_order == ["enginesize", "horsepower"]
        assert result.numeric_features == ["enginesize", "horsepower"]
        assert result.categorical_features == ["drivewheel"]
        assert result.target_column == "price"
        assert result.categorical_options == {"drivewheel": ["rwd", "fwd"]}

        args, kwargs = mock_request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == f"http://test.com{SCHEMA_PATH}"


class TestClientProtocolEnforcement:
    def test_all_errors_inherit_from_api_client_error(self):
        assert issubclass(APIConnectionError, APIClientError)
        assert issubclass(APITimeoutError, APIClientError)
        assert issubclass(APIHTTPError, APIClientError)
        assert issubclass(APIInvalidResponseError, APIClientError)

    def test_prediction_result_is_dataclass_with_prediction(self):
        import dataclasses
        assert dataclasses.is_dataclass(PredictionResult)
        fields = {f.name for f in dataclasses.fields(PredictionResult)}
        assert "prediction" in fields
        assert "currency" in fields
        assert "model_name" in fields

    def test_no_array_access_in_response_parsing(self):
        import inspect
        source = inspect.getsource(PredictionAPIClient.predict)
        assert "[0]" not in source, "Client must not use array indexing on response"
        assert "[1]" not in source
        assert "predictions[" not in source

    def test_error_response_parsing_uses_dict_keys(self):
        import inspect
        source = inspect.getsource(PredictionAPIClient._parse_error_response)
        assert ".get(\"error\"" in source or ".get('error'" in source
        assert ".get(\"detail\"" in source or ".get('detail'" in source

    def test_no_hardcoded_urls_in_client(self):
        import inspect
        source = inspect.getsource(PredictionAPIClient)
        assert "herokuapp.com" not in source, "No hardcoded production domains"
        assert "carpriceapi" not in source

    def test_client_does_not_assume_list_response(self):
        import inspect
        source = inspect.getsource(PredictionAPIClient.predict)
        assert "isinstance(data, dict)" in source, "Client must verify response is dict"
