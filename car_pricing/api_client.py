from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests

from car_pricing.feature_schema import FEATURE_ORDER


DEFAULT_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))
DEFAULT_FASTAPI_BASE_URL = os.environ.get("FASTAPI_BASE_URL", "http://localhost:8000")
DEFAULT_BENTOML_BASE_URL = os.environ.get("BENTOML_BASE_URL", "http://localhost:3000")


DEFAULT_SCHEMA = {
    "feature_order": [
        "enginesize", "curbweight", "horsepower", "highwaympg",
        "carwidth", "wheelbase", "drivewheel", "citympg",
        "boreratio", "cylindernumber",
    ],
    "numeric_features": [
        "enginesize", "curbweight", "horsepower", "highwaympg",
        "carwidth", "wheelbase", "citympg", "boreratio",
    ],
    "categorical_features": ["drivewheel", "cylindernumber"],
    "target_column": "price",
    "categorical_options": {
        "drivewheel": [
            {"display": "Four Wheel Drive (4WD)", "form_value": "4wd", "model_code": 0},
            {"display": "Front Wheel Drive (FWD)", "form_value": "fwd", "model_code": 1},
            {"display": "Rear Wheel Drive (RWD)", "form_value": "rwd", "model_code": 2},
        ],
        "cylindernumber": [
            {"display": "2 cylinders", "form_value": "two", "model_code": 6},
            {"display": "3 cylinders", "form_value": "three", "model_code": 4},
            {"display": "4 cylinders", "form_value": "four", "model_code": 2},
            {"display": "5 cylinders", "form_value": "five", "model_code": 1},
            {"display": "6 cylinders", "form_value": "six", "model_code": 3},
            {"display": "8 cylinders", "form_value": "eight", "model_code": 0},
            {"display": "12 cylinders", "form_value": "twelve", "model_code": 5},
        ],
    },
}

FIELD_DISPLAY_NAMES = {
    "enginesize": "Engine Size",
    "curbweight": "Curb Weight",
    "horsepower": "Horsepower",
    "highwaympg": "Highway Miles Per Gallon",
    "carwidth": "Car Width",
    "wheelbase": "Wheel Base",
    "drivewheel": "Drive Wheel",
    "citympg": "City Miles Per Gallon",
    "boreratio": "Bore Ratio",
    "cylindernumber": "Number of Cylinders",
}


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
        return self._request("GET", "/schema")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/predict", json=features)

    def predict_batch(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._request("POST", "/predict_batch", json={"rows": rows})

    def get_metadata(self) -> Dict[str, Any]:
        return self._request("GET", "/metadata")

    def get_status(self) -> Dict[str, Any]:
        return self._request("GET", "/status")


class BentoMLClient(BasePredictionClient):
    service_type = ServiceType.BENTOML

    def get_schema(self) -> Dict[str, Any]:
        return self._request("POST", "/schema", data="")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        ordered = {f: features[f] for f in FEATURE_ORDER}
        result = self._request("POST", "/predict_batch", json={"rows": [ordered]})
        if isinstance(result, dict) and "predictions" in result and result["predictions"]:
            return {"prediction": result["predictions"][0], "status": "ok"}
        return result

    def predict_batch(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        ordered_rows = [{f: row[f] for f in FEATURE_ORDER} for row in rows]
        return self._request("POST", "/predict_batch", json={"rows": ordered_rows})

    def get_metadata(self) -> Dict[str, Any]:
        return self._request("POST", "/metadata", data="")

    def get_status(self) -> Dict[str, Any]:
        return self._request("POST", "/status", data="")


def create_client(
    service_type: Union[str, ServiceType] = ServiceType.FASTAPI,
    base_url: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    **kwargs,
) -> BasePredictionClient:
    if isinstance(service_type, str):
        service_type = ServiceType(service_type.lower())

    if base_url is None:
        if service_type == ServiceType.FASTAPI:
            base_url = DEFAULT_FASTAPI_BASE_URL
        elif service_type == ServiceType.BENTOML:
            base_url = DEFAULT_BENTOML_BASE_URL
        else:
            raise ValueError(f"Unknown service type: {service_type}")

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
