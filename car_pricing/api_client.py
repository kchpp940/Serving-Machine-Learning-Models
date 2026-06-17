from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import requests

from car_pricing.feature_schema import (
    FeatureSchema,
    FIELD_DISPLAY_NAMES,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    WORD_TO_NUM_CYLINDERS,
    DRIVEWheel_DISPLAY,
)


@dataclass
class ApiError:
    message: str
    source: str = "unknown"
    status_code: Optional[int] = None
    detail: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def display(self) -> str:
        parts = [self.message]
        if self.detail:
            parts.append(self.detail)
        if self.status_code is not None:
            parts.append(f"(status={self.status_code})")
        return " ".join(p for p in parts if p)

    def __str__(self) -> str:
        return self.display()


def _display_name(field_name: str, raw_class: str) -> str:
    if field_name == "drivewheel":
        return DRIVEWheel_DISPLAY.get(raw_class, raw_class.upper())
    if field_name == "cylindernumber":
        num = WORD_TO_NUM_CYLINDERS.get(raw_class.lower())
        if num is not None:
            return f"{num} cylinders"
        return raw_class
    return raw_class


class CarPricingClient:
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None):
        self._base_url = (base_url or os.environ.get("API_BASE_URL", "http://localhost:8000")).rstrip("/")
        self._timeout = int(timeout or os.environ.get("API_REQUEST_TIMEOUT", "10"))
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def timeout(self) -> int:
        return self._timeout

    def _url(self, endpoint: str) -> str:
        return f"{self._base_url}/{endpoint.lstrip('/')}"

    def fetch_schema(self) -> Tuple[Optional[Dict[str, Any]], Optional[ApiError]]:
        try:
            resp = self._session.get(self._url("/schema"), timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            required = ["feature_order", "numeric_features", "categorical_features", "categorical_options"]
            missing = [k for k in required if k not in data]
            if missing:
                return None, ApiError(
                    message="Schema response is incomplete",
                    source="validation",
                    status_code=resp.status_code,
                    detail=f"Missing fields: {', '.join(missing)}",
                    context={"endpoint": "/schema", "missing": missing},
                )
            return data, None
        except requests.exceptions.ConnectionError as e:
            return None, ApiError(
                message="Unable to connect to the prediction service to fetch schema.",
                source="connection",
                detail=str(e),
                context={"endpoint": "/schema", "base_url": self._base_url},
            )
        except requests.exceptions.Timeout as e:
            return None, ApiError(
                message="Schema request timed out.",
                source="timeout",
                detail=str(e),
                context={"endpoint": "/schema", "timeout": self._timeout},
            )
        except requests.exceptions.HTTPError as e:
            detail = None
            try:
                detail = resp.json().get("detail")
            except Exception:
                pass
            return None, ApiError(
                message="Server returned error when fetching schema.",
                source="http",
                status_code=resp.status_code,
                detail=detail or str(e),
                context={"endpoint": "/schema", "status_code": resp.status_code},
            )
        except ValueError as e:
            return None, ApiError(
                message="Schema response was not valid JSON.",
                source="json",
                detail=str(e),
                context={"endpoint": "/schema"},
            )
        except Exception as e:
            return None, ApiError(
                message="Unexpected error fetching schema.",
                source="unknown",
                detail=str(e),
                context={"endpoint": "/schema", "exception_type": type(e).__name__},
            )

    def predict(self, values: Dict[str, Any]) -> Tuple[Optional[float], Optional[ApiError]]:
        try:
            resp = self._session.post(self._url("/predict"), json=values, timeout=self._timeout)
            resp.raise_for_status()
            body = resp.json()
            prediction = body.get("prediction")
            if prediction is None:
                return None, ApiError(
                    message="Unexpected response format from server.",
                    source="validation",
                    status_code=resp.status_code,
                    detail=f"Response body missing 'prediction' field: {body}",
                    context={"endpoint": "/predict"},
                )
            return float(prediction), None
        except requests.exceptions.ConnectionError as e:
            return None, ApiError(
                message="Unable to connect to the prediction service. Please check that the API server is running.",
                source="connection",
                detail=str(e),
                context={"endpoint": "/predict", "base_url": self._base_url},
            )
        except requests.exceptions.Timeout as e:
            return None, ApiError(
                message="The request to the prediction service timed out. Please try again later.",
                source="timeout",
                detail=str(e),
                context={"endpoint": "/predict", "timeout": self._timeout},
            )
        except requests.exceptions.HTTPError as e:
            detail = None
            try:
                detail = resp.json().get("detail")
            except Exception:
                pass
            return None, ApiError(
                message="Server returned an error.",
                source="http",
                status_code=resp.status_code,
                detail=detail or str(e),
                context={"endpoint": "/predict", "status_code": resp.status_code},
            )
        except ValueError as e:
            return None, ApiError(
                message="The server returned an invalid response. Please try again later.",
                source="json",
                detail=str(e),
                context={"endpoint": "/predict"},
            )
        except Exception as e:
            return None, ApiError(
                message="An unexpected error occurred.",
                source="unknown",
                detail=str(e),
                context={"endpoint": "/predict", "exception_type": type(e).__name__},
            )

    def health(self) -> Tuple[Optional[Dict[str, Any]], Optional[ApiError]]:
        try:
            resp = self._session.get(self._url("/health"), timeout=self._timeout)
            resp.raise_for_status()
            return resp.json(), None
        except requests.exceptions.ConnectionError as e:
            return None, ApiError(message="Connection failed.", source="connection", detail=str(e), context={"endpoint": "/health"})
        except requests.exceptions.Timeout as e:
            return None, ApiError(message="Request timed out.", source="timeout", detail=str(e), context={"endpoint": "/health"})
        except requests.exceptions.HTTPError as e:
            return None, ApiError(message="Server returned error.", source="http", status_code=resp.status_code, detail=str(e), context={"endpoint": "/health"})
        except Exception as e:
            return None, ApiError(message="Unexpected error.", source="unknown", detail=str(e), context={"endpoint": "/health", "exception_type": type(e).__name__})

    def metadata(self) -> Tuple[Optional[Dict[str, Any]], Optional[ApiError]]:
        try:
            resp = self._session.get(self._url("/metadata"), timeout=self._timeout)
            resp.raise_for_status()
            return resp.json(), None
        except requests.exceptions.ConnectionError as e:
            return None, ApiError(message="Connection failed.", source="connection", detail=str(e), context={"endpoint": "/metadata"})
        except requests.exceptions.Timeout as e:
            return None, ApiError(message="Request timed out.", source="timeout", detail=str(e), context={"endpoint": "/metadata"})
        except requests.exceptions.HTTPError as e:
            return None, ApiError(message="Server returned error.", source="http", status_code=resp.status_code, detail=str(e), context={"endpoint": "/metadata"})
        except Exception as e:
            return None, ApiError(message="Unexpected error.", source="unknown", detail=str(e), context={"endpoint": "/metadata", "exception_type": type(e).__name__})

    def status(self) -> Tuple[Optional[Dict[str, Any]], Optional[ApiError]]:
        try:
            resp = self._session.get(self._url("/status"), timeout=self._timeout)
            resp.raise_for_status()
            return resp.json(), None
        except requests.exceptions.ConnectionError as e:
            return None, ApiError(message="Connection failed.", source="connection", detail=str(e), context={"endpoint": "/status"})
        except requests.exceptions.Timeout as e:
            return None, ApiError(message="Request timed out.", source="timeout", detail=str(e), context={"endpoint": "/status"})
        except requests.exceptions.HTTPError as e:
            return None, ApiError(message="Server returned error.", source="http", status_code=resp.status_code, detail=str(e), context={"endpoint": "/status"})
        except Exception as e:
            return None, ApiError(message="Unexpected error.", source="unknown", detail=str(e), context={"endpoint": "/status", "exception_type": type(e).__name__})

    @staticmethod
    def build_fallback_schema() -> Dict[str, Any]:
        schema = FeatureSchema.default()
        categorical_options: Dict[str, list] = {}
        for f in schema.categorical_features:
            if f == "drivewheel":
                categorical_options[f] = [
                    {"display": _display_name(f, cls), "form_value": cls}
                    for cls in ("fwd", "rwd", "4wd")
                ]
            elif f == "cylindernumber":
                categorical_options[f] = [
                    {"display": _display_name(f, cls), "form_value": cls}
                    for cls in sorted(WORD_TO_NUM_CYLINDERS.keys())
                ]
            else:
                categorical_options[f] = []
        return {
            "feature_order": list(schema.feature_order),
            "numeric_features": list(schema.numeric_features),
            "categorical_features": list(schema.categorical_features),
            "target_column": schema.target_column,
            "categorical_options": categorical_options,
        }
