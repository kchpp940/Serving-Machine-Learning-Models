from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from car_pricing.feature_schema import (
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    FIELD_DISPLAY_NAMES,
    FIELD_DEFAULT_VALUES,
    _display_name,
)


DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 10


def _build_default_categorical_options() -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    try:
        from sklearn.preprocessing import LabelEncoder

        legacy_classes = {
            "drivewheel": ["4wd", "fwd", "rwd"],
            "cylindernumber": ["eight", "five", "four", "six", "three", "twelve", "two"],
        }
        for col, classes in legacy_classes.items():
            le = LabelEncoder()
            le.fit([str(c) for c in classes])
            codes = le.transform(le.classes_)
            options = []
            for raw, code in zip(le.classes_, codes):
                options.append({
                    "display": _display_name(col, raw),
                    "form_value": raw,
                    "model_code": int(code),
                })
            result[col] = options
    except ImportError:
        result = {
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
        }
    return result


def build_default_schema() -> Dict[str, Any]:
    return {
        "feature_order": list(FEATURE_ORDER),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "categorical_options": _build_default_categorical_options(),
        "display_names": dict(FIELD_DISPLAY_NAMES),
        "default_values": dict(FIELD_DEFAULT_VALUES),
    }


@dataclass
class ApiError:
    message: str
    kind: str = "unknown"
    status_code: Optional[int] = None
    detail: Optional[str] = None

    def display(self) -> str:
        if self.detail:
            return f"{self.message}: {self.detail}"
        return self.message

    @classmethod
    def connection_error(cls, msg: str = "Unable to connect to the prediction service.") -> "ApiError":
        return cls(message=msg, kind="connection")

    @classmethod
    def timeout_error(cls, msg: str = "Request timed out.") -> "ApiError":
        return cls(message=msg, kind="timeout")

    @classmethod
    def http_error(cls, status_code: int, detail: str = "", msg: str = "") -> "ApiError":
        message = msg or f"Server returned error ({status_code})"
        return cls(message=message, kind="http", status_code=status_code, detail=detail)

    @classmethod
    def parse_error(cls, msg: str = "Response was not valid JSON.") -> "ApiError":
        return cls(message=msg, kind="parse")

    @classmethod
    def validation_error(cls, msg: str) -> "ApiError":
        return cls(message=msg, kind="validation")

    @classmethod
    def unexpected_error(cls, exc: Exception) -> "ApiError":
        return cls(message=f"Unexpected error: {str(exc)}", kind="unexpected")


@dataclass
class HealthStatus:
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    schema_version: Optional[str] = None
    data_version: Optional[str] = None
    n_features: Optional[int] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthStatus":
        return cls(
            status=str(data.get("status", "unknown")),
            model_loaded=bool(data.get("model_loaded", False)),
            model_name=data.get("model_name"),
            model_type=data.get("model_type"),
            schema_version=data.get("schema_version"),
            data_version=data.get("data_version"),
            n_features=data.get("n_features"),
            error=data.get("error"),
        )

    def is_healthy(self) -> bool:
        return self.status.lower() == "healthy" and self.model_loaded


@dataclass
class PredictionResult:
    prediction: float
    values: Dict[str, Any] = field(default_factory=dict)
    scenario_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionResult":
        return cls(
            prediction=float(data["prediction"]),
            values=data.get("values", {}),
            scenario_name=data.get("scenario_name"),
        )


@dataclass
class BatchPredictionResult:
    results: List[PredictionResult] = field(default_factory=list)

    def predictions(self) -> List[float]:
        return [r.prediction for r in self.results]

    def __len__(self) -> int:
        return len(self.results)


@dataclass
class ServiceMetadata:
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def flat_sections(self) -> List[Tuple[str, Dict[str, Any]]]:
        top: Dict[str, Any] = {}
        nested: List[Tuple[str, Dict[str, Any]]] = []
        for key, value in self.raw.items():
            if isinstance(value, dict):
                nested.append((key.replace("_", " ").title(), value))
            else:
                top[key] = value
        if top:
            nested.insert(0, ("Overview", top))
        return nested

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceMetadata":
        return cls(raw=data)


class CarPricingApiClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)).rstrip("/")
        self.timeout = timeout or int(os.environ.get("API_REQUEST_TIMEOUT", str(DEFAULT_TIMEOUT)))

    def _get(self, path: str) -> Tuple[Optional[Dict[str, Any]], Optional[ApiError]]:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json(), None
        except requests.exceptions.ConnectionError:
            return None, ApiError.connection_error()
        except requests.exceptions.Timeout:
            return None, ApiError.timeout_error()
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                pass
            return None, ApiError.http_error(resp.status_code, detail=detail, msg=str(e))
        except ValueError:
            return None, ApiError.parse_error()
        except Exception as e:
            return None, ApiError.unexpected_error(e)

    def _post(self, path: str, payload: Any) -> Tuple[Optional[Dict[str, Any]], Optional[ApiError]]:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json(), None
        except requests.exceptions.ConnectionError:
            return None, ApiError.connection_error()
        except requests.exceptions.Timeout:
            return None, ApiError.timeout_error()
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                pass
            return None, ApiError.http_error(resp.status_code, detail=detail, msg=str(e))
        except ValueError:
            return None, ApiError.parse_error()
        except Exception as e:
            return None, ApiError.unexpected_error(e)

    def fetch_schema(self) -> Tuple[Dict[str, Any], Optional[ApiError]]:
        data, err = self._get("/schema")
        if err:
            return build_default_schema(), err

        required = ["feature_order", "numeric_features", "categorical_features", "categorical_options"]
        missing = [k for k in required if k not in data]
        if missing:
            msg = f"Schema missing fields: {', '.join(missing)}"
            return build_default_schema(), ApiError.validation_error(msg)

        if "display_names" not in data:
            data["display_names"] = dict(FIELD_DISPLAY_NAMES)
        if "default_values" not in data:
            data["default_values"] = dict(FIELD_DEFAULT_VALUES)
        if "target_column" not in data:
            data["target_column"] = TARGET_COLUMN

        return data, None

    def predict(self, values: Dict[str, Any]) -> Tuple[Optional[float], Optional[ApiError]]:
        data, err = self._post("/predict", values)
        if err:
            return None, err
        prediction = data.get("prediction")
        if prediction is None:
            return None, ApiError.validation_error(f"Unexpected response format from server: {data}")
        return float(prediction), None

    def predict_batch(
        self,
        batch: List[Dict[str, Any]],
        scenario_names: Optional[List[str]] = None,
    ) -> Tuple[Optional[BatchPredictionResult], Optional[ApiError]]:
        if not batch:
            return None, ApiError.validation_error("Empty batch provided.")

        results: List[PredictionResult] = []

        for i, values in enumerate(batch):
            name = scenario_names[i] if scenario_names and i < len(scenario_names) else None
            prediction, err = self.predict(values)
            if err:
                return None, ApiError(
                    message=f"Batch prediction failed at index {i}",
                    kind=err.kind,
                    status_code=err.status_code,
                    detail=err.display(),
                )
            results.append(PredictionResult(
                prediction=prediction,
                values=values,
                scenario_name=name,
            ))

        return BatchPredictionResult(results=results), None

    def get_health(self) -> Tuple[Optional[HealthStatus], Optional[ApiError]]:
        data, err = self._get("/health")
        if err:
            return None, err
        return HealthStatus.from_dict(data), None

    def get_status(self) -> Tuple[Optional[ServiceMetadata], Optional[ApiError]]:
        data, err = self._get("/status")
        if err:
            return None, err
        return ServiceMetadata.from_dict(data), None

    def get_metadata(self) -> Tuple[Optional[ServiceMetadata], Optional[ApiError]]:
        data, err = self._get("/metadata")
        if err:
            return None, err
        return ServiceMetadata.from_dict(data), None
