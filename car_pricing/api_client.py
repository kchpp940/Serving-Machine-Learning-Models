from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests as re


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))


@dataclass
class ApiError:
    message: str
    status_code: Optional[int] = None
    detail: Optional[str] = None

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message}: {self.detail}"
        return self.message


def _base_url() -> str:
    return API_BASE_URL.rstrip("/")


def _handle_error(label: str, resp: Optional[re.Response] = None, exc: Optional[Exception] = None) -> ApiError:
    if isinstance(exc, re.exceptions.ConnectionError):
        return ApiError(f"Unable to connect to the prediction service{(' ' + label) if label else ''}.")
    if isinstance(exc, re.exceptions.Timeout):
        return ApiError(f"{label or 'Request'} timed out.")
    if isinstance(exc, re.exceptions.HTTPError):
        detail = None
        status = getattr(resp, "status_code", None)
        if resp is not None:
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                pass
        return ApiError(f"Server returned an error{(' (' + str(status) + ')') if status else ''}", status_code=status, detail=detail)
    if isinstance(exc, ValueError):
        return ApiError("The server returned an invalid response.")
    if exc is not None:
        return ApiError(f"An unexpected error occurred: {str(exc)}")
    return ApiError(f"Unknown error{(' ' + label) if label else ''}.")


def fetch_schema() -> Tuple[Optional[Dict[str, Any]], Optional[ApiError]]:
    url = f"{_base_url()}/schema"
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        required = ["feature_order", "numeric_features", "categorical_features", "categorical_options"]
        missing = [k for k in required if k not in data]
        if missing:
            return None, ApiError(f"Schema missing fields: {', '.join(missing)}")
        return data, None
    except Exception as e:
        return None, _handle_error("fetching schema", resp=resp if isinstance(e, re.exceptions.HTTPError) else None, exc=e)


def predict(values: Dict[str, Any]) -> Tuple[Optional[float], Optional[ApiError]]:
    url = f"{_base_url()}/predict"
    try:
        resp = re.post(url, json=values, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        prediction = body.get("prediction")
        if prediction is None:
            return None, ApiError("Unexpected response format from server", detail=str(body))
        return float(prediction), None
    except Exception as e:
        return None, _handle_error("prediction", resp=resp if isinstance(e, re.exceptions.HTTPError) else None, exc=e)


def fetch_health() -> Tuple[Optional[Dict[str, Any]], Optional[ApiError]]:
    url = f"{_base_url()}/health"
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, _handle_error("health check", resp=resp if isinstance(e, re.exceptions.HTTPError) else None, exc=e)


def fetch_status() -> Tuple[Optional[Dict[str, Any]], Optional[ApiError]]:
    url = f"{_base_url()}/status"
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, _handle_error("status", resp=resp if isinstance(e, re.exceptions.HTTPError) else None, exc=e)


def fetch_metadata() -> Tuple[Optional[Dict[str, Any]], Optional[ApiError]]:
    url = f"{_base_url()}/metadata"
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, _handle_error("metadata", resp=resp if isinstance(e, re.exceptions.HTTPError) else None, exc=e)
