from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

try:
    import requests
except ImportError:
    requests = None

ENV_API_BASE_URL = "API_BASE_URL"
ENV_API_TIMEOUT = "API_REQUEST_TIMEOUT"

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 10

HEALTH_PATH = "/health"
PREDICT_PATH = "/predict"
SCHEMA_PATH = "/schema"


class APIClientError(Exception):
    pass


class APIConnectionError(APIClientError):
    pass


class APITimeoutError(APIClientError):
    pass


class APIHTTPError(APIClientError):
    def __init__(self, status_code: int, error_type: str, detail: str):
        self.status_code = status_code
        self.error_type = error_type
        self.detail = detail
        super().__init__(f"HTTP {status_code} [{error_type}]: {detail}")


class APIInvalidResponseError(APIClientError):
    pass


@dataclass
class PredictionResult:
    prediction: float
    currency: str = "USD"
    model_name: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class HealthResult:
    status: str
    service: Optional[str] = None
    version: Optional[str] = None
    model_loaded: Optional[bool] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class SchemaResult:
    feature_order: list
    numeric_features: list
    categorical_features: list
    target_column: str
    categorical_options: Dict[str, list]
    raw_response: Optional[Dict[str, Any]] = None


class PredictionAPIClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        verify: Union[bool, str] = True,
    ):
        if requests is None:
            raise RuntimeError("requests library is required. Install with: pip install requests")

        self.base_url = (base_url or os.environ.get(ENV_API_BASE_URL, DEFAULT_API_BASE_URL)).rstrip("/")
        self.timeout = int(timeout or os.environ.get(ENV_API_TIMEOUT, str(DEFAULT_TIMEOUT)))
        self.verify = verify

    def _request(self, method: str, path: str, json: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method=method,
                url=url,
                json=json,
                timeout=self.timeout,
                verify=self.verify,
            )
        except requests.exceptions.ConnectionError as e:
            raise APIConnectionError(f"Cannot connect to {self.base_url}. Check network and API URL.") from e
        except requests.exceptions.Timeout as e:
            raise APITimeoutError(f"Request to {url} timed out after {self.timeout}s.") from e
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"Request failed: {str(e)}") from e

        if response.status_code >= 400:
            self._parse_error_response(response)

        try:
            return response.json()
        except ValueError as e:
            raise APIInvalidResponseError(f"Server returned invalid JSON: {str(e)}") from e

    def _parse_error_response(self, response) -> None:
        status_code = response.status_code
        error_type = f"HTTP{status_code}"
        detail = f"HTTP Error {status_code}"

        try:
            err_data = response.json()
            if isinstance(err_data, dict):
                error_type = err_data.get("error", error_type)
                detail = err_data.get("detail", detail)
        except (ValueError, AttributeError):
            pass

        raise APIHTTPError(
            status_code=status_code,
            error_type=error_type,
            detail=detail,
        )

    def health_check(self) -> HealthResult:
        data = self._request("GET", HEALTH_PATH)

        if not isinstance(data, dict):
            raise APIInvalidResponseError("Health response is not a JSON object")

        status = data.get("status")
        if status is None:
            raise APIInvalidResponseError("Health response missing 'status' field")

        return HealthResult(
            status=status,
            service=data.get("service"),
            version=data.get("version"),
            model_loaded=data.get("model_loaded"),
            raw_response=data,
        )

    def get_schema(self) -> SchemaResult:
        data = self._request("GET", SCHEMA_PATH)

        if not isinstance(data, dict):
            raise APIInvalidResponseError("Schema response is not a JSON object")

        required = ["feature_order", "numeric_features", "categorical_features", "categorical_options", "target_column"]
        missing = [k for k in required if k not in data]
        if missing:
            raise APIInvalidResponseError(f"Schema response missing fields: {', '.join(missing)}")

        return SchemaResult(
            feature_order=list(data["feature_order"]),
            numeric_features=list(data["numeric_features"]),
            categorical_features=list(data["categorical_features"]),
            target_column=data["target_column"],
            categorical_options=dict(data["categorical_options"]),
            raw_response=data,
        )

    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        data = self._request("POST", PREDICT_PATH, json=features)

        if not isinstance(data, dict):
            raise APIInvalidResponseError("Prediction response is not a JSON object")

        prediction = data.get("prediction")
        if prediction is None:
            raise APIInvalidResponseError(
                f"Prediction response missing 'prediction' field. Got: {list(data.keys())}"
            )

        try:
            prediction_value = float(prediction)
        except (ValueError, TypeError) as e:
            raise APIInvalidResponseError(
                f"'prediction' field is not a valid number: {prediction!r}"
            ) from e

        return PredictionResult(
            prediction=prediction_value,
            currency=data.get("currency", "USD"),
            model_name=data.get("model_name"),
            raw_response=data,
        )

    @staticmethod
    def format_error(error: Exception) -> str:
        if isinstance(error, APIConnectionError):
            return f"❌ Connection Error: {str(error)}"
        elif isinstance(error, APITimeoutError):
            return f"⏱️ Timeout: {str(error)}"
        elif isinstance(error, APIHTTPError):
            if error.status_code == 400:
                return f"⚠️ Invalid Input: {error.detail}"
            elif error.status_code == 404:
                return f"🔍 Not Found: {error.detail}"
            elif error.status_code == 500:
                return f"💥 Server Error: {error.detail}"
            else:
                return f"❌ HTTP Error [{error.status_code}]: {error.detail}"
        elif isinstance(error, APIInvalidResponseError):
            return f"📝 Invalid Response: {str(error)}"
        elif isinstance(error, APIClientError):
            return f"❌ API Error: {str(error)}"
        else:
            return f"❌ Unexpected Error: {str(error)}"
