from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests

from car_pricing.feature_schema import FEATURE_ORDER, CATEGORICAL_FEATURES


DEFAULT_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))
DEFAULT_FASTAPI_BASE_URL = os.environ.get("FASTAPI_BASE_URL", "http://localhost:8000")
DEFAULT_BENTOML_BASE_URL = os.environ.get("BENTOML_BASE_URL", "http://localhost:3000")


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


@dataclass
class ServiceError(Exception):
    category: ErrorCategory
    message: str
    status_code: Optional[int] = None
    raw_detail: Optional[str] = None

    def __str__(self) -> str:
        return self.message


@dataclass
class SchemaInfo:
    feature_order: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    target_column: str
    categorical_options: Dict[str, List[Dict[str, Any]]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaInfo":
        return cls(
            feature_order=list(data["feature_order"]),
            numeric_features=list(data["numeric_features"]),
            categorical_features=list(data["categorical_features"]),
            target_column=data.get("target_column", "price"),
            categorical_options=dict(data.get("categorical_options", {})),
        )


@dataclass
class PredictionResult:
    prediction: float
    status: str = "ok"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionResult":
        return cls(
            prediction=float(data["prediction"]),
            status=data.get("status", "ok"),
        )


@dataclass
class BatchPredictionResult:
    predictions: List[float]
    status: str = "ok"
    count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchPredictionResult":
        preds = data.get("predictions", data.get("prediction", []))
        if isinstance(preds, (int, float)):
            preds = [float(preds)]
        preds = [float(p) for p in preds]
        return cls(
            predictions=preds,
            status=data.get("status", "ok"),
            count=len(preds),
        )


@dataclass
class ServiceMetadata:
    service_name: str
    version: str
    model_name: Optional[str] = None
    model_mode: Optional[str] = None
    n_features: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceMetadata":
        return cls(
            service_name=data.get("service_name", data.get("name", "unknown")),
            version=data.get("version", "unknown"),
            model_name=data.get("model_name"),
            model_mode=data.get("model_mode"),
            n_features=data.get("n_features"),
            extra={k: v for k, v in data.items() if k not in {
                "service_name", "name", "version", "model_name", "model_mode", "n_features"
            }},
        )


@dataclass
class ServiceStatus:
    status: str
    uptime_seconds: Optional[float] = None
    model_loaded: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceStatus":
        return cls(
            status=data.get("status", "unknown"),
            uptime_seconds=data.get("uptime_seconds"),
            model_loaded=data.get("model_loaded", False),
            extra={k: v for k, v in data.items() if k not in {
                "status", "uptime_seconds", "model_loaded"
            }},
        )


DEFAULT_FALLBACK_SCHEMA = SchemaInfo(
    feature_order=[
        "enginesize", "curbweight", "horsepower", "highwaympg",
        "carwidth", "wheelbase", "drivewheel", "citympg",
        "boreratio", "cylindernumber",
    ],
    numeric_features=[
        "enginesize", "curbweight", "horsepower", "highwaympg",
        "carwidth", "wheelbase", "citympg", "boreratio",
    ],
    categorical_features=["drivewheel", "cylindernumber"],
    target_column="price",
    categorical_options={
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
)


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
        ErrorCategory.CONNECTION: "无法连接到预测服务，请检查服务是否启动。",
        ErrorCategory.TIMEOUT: "请求超时，请稍后重试。",
        ErrorCategory.BAD_REQUEST: f"请求参数错误: {detail or '请检查输入数据。'}",
        ErrorCategory.AUTH: "认证失败，请检查凭证。",
        ErrorCategory.NOT_FOUND: "请求的接口不存在。",
        ErrorCategory.SERVER_ERROR: f"服务端错误: {detail or '请稍后重试。'}",
        ErrorCategory.INVALID_RESPONSE: "服务返回了无效的响应格式。",
        ErrorCategory.UNKNOWN: f"未知错误: {detail or '请联系管理员。'}",
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

    def get_schema(self) -> SchemaInfo:
        raise NotImplementedError

    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        raise NotImplementedError

    def predict_batch(self, rows: List[Dict[str, Any]]) -> BatchPredictionResult:
        raise NotImplementedError

    def get_metadata(self) -> ServiceMetadata:
        raise NotImplementedError

    def get_status(self) -> ServiceStatus:
        raise NotImplementedError


class FastAPIClient(BasePredictionClient):
    service_type = ServiceType.FASTAPI

    def get_schema(self) -> SchemaInfo:
        data = self._request("GET", "/schema")
        return SchemaInfo.from_dict(data)

    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        data = self._request("POST", "/predict", json=features)
        return PredictionResult.from_dict(data)

    def predict_batch(self, rows: List[Dict[str, Any]]) -> BatchPredictionResult:
        data = self._request("POST", "/predict_batch", json={"rows": rows})
        return BatchPredictionResult.from_dict(data)

    def get_metadata(self) -> ServiceMetadata:
        data = self._request("GET", "/metadata")
        return ServiceMetadata.from_dict(data)

    def get_status(self) -> ServiceStatus:
        data = self._request("GET", "/status")
        return ServiceStatus.from_dict(data)


class BentoMLClient(BasePredictionClient):
    service_type = ServiceType.BENTOML

    def get_schema(self) -> SchemaInfo:
        data = self._request("POST", "/schema", data="")
        return SchemaInfo.from_dict(data)

    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        ordered = {f: features[f] for f in FEATURE_ORDER}
        data = self._request("POST", "/predict_json", json=ordered)
        if isinstance(data, list) and data:
            return PredictionResult(prediction=float(data[0]))
        if isinstance(data, dict) and "prediction" in data:
            return PredictionResult.from_dict(data)
        if isinstance(data, dict) and "predictions" in data:
            preds = data["predictions"]
            if isinstance(preds, list) and preds:
                return PredictionResult(prediction=float(preds[0]))
        raise ServiceError(
            category=ErrorCategory.INVALID_RESPONSE,
            message="BentoML 返回了无法解析的预测响应",
        )

    def predict_batch(self, rows: List[Dict[str, Any]]) -> BatchPredictionResult:
        ordered_rows = [{f: row[f] for f in FEATURE_ORDER} for row in rows]
        data = self._request("POST", "/predict_batch", json={"rows": ordered_rows})
        if isinstance(data, list):
            return BatchPredictionResult(
                predictions=[float(p) for p in data],
                count=len(data),
            )
        return BatchPredictionResult.from_dict(data)

    def get_metadata(self) -> ServiceMetadata:
        data = self._request("POST", "/metadata", data="")
        return ServiceMetadata.from_dict(data)

    def get_status(self) -> ServiceStatus:
        data = self._request("POST", "/status", data="")
        return ServiceStatus.from_dict(data)


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
            raise ValueError(f"未知服务类型: {service_type}")

    if service_type == ServiceType.FASTAPI:
        return FastAPIClient(base_url=base_url, timeout=timeout, **kwargs)
    elif service_type == ServiceType.BENTOML:
        return BentoMLClient(base_url=base_url, timeout=timeout, **kwargs)
    else:
        raise ValueError(f"未知服务类型: {service_type}")


__all__ = [
    "ServiceType",
    "ErrorCategory",
    "ServiceError",
    "SchemaInfo",
    "PredictionResult",
    "BatchPredictionResult",
    "ServiceMetadata",
    "ServiceStatus",
    "BasePredictionClient",
    "FastAPIClient",
    "BentoMLClient",
    "create_client",
    "DEFAULT_TIMEOUT",
    "DEFAULT_FASTAPI_BASE_URL",
    "DEFAULT_BENTOML_BASE_URL",
    "DEFAULT_FALLBACK_SCHEMA",
]
