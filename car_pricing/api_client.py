from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests

from car_pricing.feature_schema import FEATURE_ORDER, default_schema_dict, field_display_names


DEFAULT_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))
DEFAULT_FASTAPI_BASE_URL = os.environ.get("FASTAPI_BASE_URL", "http://localhost:8000")
DEFAULT_BENTOML_BASE_URL = os.environ.get("BENTOML_BASE_URL", "http://localhost:3000")


DEFAULT_SCHEMA = default_schema_dict()
FIELD_DISPLAY_NAMES = field_display_names()


class ServiceType(str, Enum):
    FASTAPI = "fastapi"
    BENTOML = "bentoml"


class ErrorCategory(str, Enum):
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    BAD_REQUEST = "bad_request"
    SERVER_ERROR = "server_error"
    AUTH = "auth"
    UNKNOWN = "unknown"
    INVALID_RESPONSE = "invalid_response"
    NOT_FOUND = "not_found"


class ServiceError(Exception):
    def __init__(
        self,
        category: ErrorCategory,
        message: str,
        status_code: Optional[int] = None,
        raw_detail: Optional[str] = None,
    ):
        super().__init__(message)
        self.category = category
        self.message = message
        self.status_code = status_code
        self.raw_detail = raw_detail

    def __str__(self) -> str:
        return self.message


def _classify_error(
    exc: Optional[Exception],
    status_code: Optional[int] = None,
) -> ErrorCategory:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return ErrorCategory.CONNECTION
    if isinstance(exc, requests.exceptions.Timeout):
        return ErrorCategory.TIMEOUT
    if status_code is not None:
        if status_code == 400 or status_code == 422:
            return ErrorCategory.BAD_REQUEST
        if status_code == 401 or status_code == 403:
            return ErrorCategory.AUTH
        if status_code == 404:
            return ErrorCategory.NOT_FOUND
        if 500 <= status_code < 600:
            return ErrorCategory.SERVER_ERROR
    if isinstance(exc, ValueError):
        return ErrorCategory.INVALID_RESPONSE
    return ErrorCategory.UNKNOWN


def _build_error_message(category: ErrorCategory, detail: str = "") -> str:
    messages = {
        ErrorCategory.CONNECTION: "Unable to connect to the prediction service.",
        ErrorCategory.TIMEOUT: "The request timed out.",
        ErrorCategory.BAD_REQUEST: f"Bad request: {detail or 'please check your input.'}",
        ErrorCategory.AUTH: "Authentication failed.",
        ErrorCategory.NOT_FOUND: "Endpoint not found.",
        ErrorCategory.SERVER_ERROR: f"Server error: {detail or 'please try again later.'}",
        ErrorCategory.INVALID_RESPONSE: "Server returned an invalid response.",
        ErrorCategory.UNKNOWN: f"Unknown error: {detail or 'please contact support.'}",
    }
    return messages.get(category, messages[ErrorCategory.UNKNOWN])


def _extract_detail(resp: Optional[requests.Response]) -> str:
    if resp is None:
        return ""
    try:
        data = resp.json()
        if isinstance(data, dict):
            return str(data.get("detail", data.get("message", data.get("error", ""))))
    except Exception:
        pass
    try:
        text = resp.text.strip()
        if text and len(text) < 500:
            return text
    except Exception:
        pass
    return ""


_SCHEMA_REQUIRED_FIELDS = [
    "feature_order", "numeric_features",
    "categorical_features", "categorical_options",
]


def _validate_schema_payload(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"Schema should be a dict, got {type(data).__name__}")
    missing = [k for k in _SCHEMA_REQUIRED_FIELDS if k not in data]
    if missing:
        raise ValueError(f"Schema missing required fields: {', '.join(missing)}")
    return data


def _validate_prediction_payload(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"Prediction response should be a dict, got {type(data).__name__}")
    return data


class BasePredictionClient:
    service_type: ServiceType

    def __init__(
        self,
        base_url: str,
        timeout: int = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.verify_ssl = verify_ssl

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("verify", self.verify_ssl)
        resp = None
        try:
            resp = self.session.request(method, self._url(path), **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            detail = _extract_detail(resp)
            status_code = resp.status_code if resp is not None else None
            category = _classify_error(exc, status_code)
            message = _build_error_message(category, detail)
            raise ServiceError(
                category=category,
                message=message,
                status_code=status_code,
                raw_detail=detail,
            ) from exc

    def _wrap_invalid(self, validator_fn, data):
        try:
            return validator_fn(data)
        except ServiceError:
            raise
        except ValueError as e:
            raise ServiceError(
                category=ErrorCategory.INVALID_RESPONSE,
                message=_build_error_message(ErrorCategory.INVALID_RESPONSE, str(e)),
                raw_detail=str(e),
            ) from e

    def get_schema(self) -> Dict[str, Any]:
        raise NotImplementedError

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def predict_batch(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def get_metadata(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_status(self) -> Dict[str, Any]:
        raise NotImplementedError


class FastAPIClient(BasePredictionClient):
    service_type = ServiceType.FASTAPI

    def get_schema(self) -> Dict[str, Any]:
        data = self._request("GET", "/schema")
        return self._wrap_invalid(_validate_schema_payload, data)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        data = self._request("POST", "/predict", json=features)
        return self._wrap_invalid(_validate_prediction_payload, data)

    def predict_batch(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        data = self._request("POST", "/predict_batch", json={"rows": rows})
        return self._wrap_invalid(_validate_prediction_payload, data)

    def get_metadata(self) -> Dict[str, Any]:
        return self._request("GET", "/metadata")

    def get_status(self) -> Dict[str, Any]:
        return self._request("GET", "/status")


class BentoMLClient(BasePredictionClient):
    service_type = ServiceType.BENTOML

    def get_schema(self) -> Dict[str, Any]:
        data = self._request("POST", "/schema", data="")
        return self._wrap_invalid(_validate_schema_payload, data)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        ordered = {f: features[f] for f in FEATURE_ORDER}
        result = self._request("POST", "/predict_batch", json={"rows": [ordered]})
        if isinstance(result, dict) and "predictions" in result and result["predictions"]:
            result = {"prediction": result["predictions"][0], "status": "ok"}
        return self._wrap_invalid(_validate_prediction_payload, result)

    def predict_batch(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        ordered_rows = [{f: row[f] for f in FEATURE_ORDER} for row in rows]
        data = self._request("POST", "/predict_batch", json={"rows": ordered_rows})
        return self._wrap_invalid(_validate_prediction_payload, data)

    def get_metadata(self) -> Dict[str, Any]:
        return self._request("POST", "/metadata", data="")

    def get_status(self) -> Dict[str, Any]:
        return self._request("POST", "/status", data="")


def create_client(
    service_type: Optional[Union[str, ServiceType]] = None,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
    **kwargs,
) -> BasePredictionClient:
    if service_type is None:
        service_type = os.environ.get("API_SERVICE_TYPE", "fastapi")
    if isinstance(service_type, str):
        service_type = ServiceType(service_type.lower())

    if base_url is None:
        base_url = os.environ.get("API_BASE_URL")

    if base_url is None:
        if service_type == ServiceType.FASTAPI:
            base_url = DEFAULT_FASTAPI_BASE_URL
        elif service_type == ServiceType.BENTOML:
            base_url = DEFAULT_BENTOML_BASE_URL
        else:
            raise ValueError(f"Unknown service type: {service_type}")

    if timeout is None:
        timeout_str = os.environ.get("API_REQUEST_TIMEOUT")
        if timeout_str is not None:
            timeout = int(timeout_str)
        else:
            timeout = DEFAULT_TIMEOUT

    if service_type == ServiceType.FASTAPI:
        return FastAPIClient(base_url=base_url, timeout=timeout, **kwargs)
    elif service_type == ServiceType.BENTOML:
        return BentoMLClient(base_url=base_url, timeout=timeout, **kwargs)
    else:
        raise ValueError(f"Unknown service type: {service_type}")


__all__ = [
    "ServiceType",
    "ErrorCategory",
    "ServiceError",
    "BasePredictionClient",
    "FastAPIClient",
    "BentoMLClient",
    "create_client",
    "DEFAULT_TIMEOUT",
    "DEFAULT_FASTAPI_BASE_URL",
    "DEFAULT_BENTOML_BASE_URL",
    "DEFAULT_SCHEMA",
    "FIELD_DISPLAY_NAMES",
]
