from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from car_pricing.feature_schema import (
    FeatureSchema,
)
from car_pricing.model_runtime import CarPriceModel
from car_pricing.api_client import (
    API_VERSION,
    detect_api_base,
)
from car_pricing.versioning import (
    compute_schema_version,
    compute_data_version,
)


CONSISTENCY_OK = "ok"
CONSISTENCY_WARNING = "warning"
CONSISTENCY_ERROR = "error"

COMPARISON_FIELDS = [
    "api_version",
    "model_mode",
    "schema_version",
    "data_version",
    "n_features",
    "feature_order",
]


@dataclass
class ServiceStatus:
    api_version: str
    service_type: str
    model_loaded: bool
    model_mode: Optional[str]
    schema_version: str
    data_version: str
    n_features: Optional[int]
    feature_order: Optional[List[str]]
    api_base_suggestion: str
    last_self_check: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _self_check_none_model() -> Dict[str, Any]:
    import time

    return {
        "passed": False,
        "checks": [
            {
                "name": "model_loaded",
                "passed": False,
                "detail": "CarPriceModel instance is None",
            }
        ],
        "errors": ["Model not loaded"],
        "timestamp": time.time(),
    }


def build_service_status(
    service_type: str,
    model: Optional[CarPriceModel],
    data_csv_path: Optional[str] = None,
) -> ServiceStatus:
    model_loaded = model is not None
    model_mode = model.mode if model is not None else None

    if model is not None and model.schema is not None:
        schema_version = compute_schema_version(model.schema)
        n_features = model.schema.n_features()
        feature_order = list(model.schema.feature_order)
        self_check_result = model.self_check()
    else:
        schema_version = "unknown"
        n_features = None
        feature_order = None
        self_check_result = _self_check_none_model()

    data_version = compute_data_version(data_csv_path)

    return ServiceStatus(
        api_version=API_VERSION,
        service_type=service_type,
        model_loaded=model_loaded,
        model_mode=model_mode,
        schema_version=schema_version,
        data_version=data_version,
        n_features=n_features,
        feature_order=feature_order,
        api_base_suggestion=detect_api_base(),
        last_self_check=self_check_result,
    )


@dataclass
class ConsistencyDiff:
    field: str
    fastapi_value: Any
    bentoml_value: Any
    severity: str  # "warning" | "error"
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ServiceErrors:
    service_type: str
    self_check_errors: List[str] = field(default_factory=list)
    endpoint_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsistencyReport:
    level: str  # "ok" | "warning" | "error"
    summary: str
    diffs: List[ConsistencyDiff] = field(default_factory=list)
    service_errors: List[ServiceErrors] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "summary": self.summary,
            "diffs": [d.to_dict() for d in self.diffs],
            "service_errors": [se.to_dict() for se in self.service_errors],
        }


def _format_value(v: Any) -> str:
    if v is None:
        return "(none)"
    if isinstance(v, list):
        return str(v)
    return str(v)


def compare_service_statuses(
    fastapi_status: Optional[Dict[str, Any]],
    bentoml_status: Optional[Dict[str, Any]],
    fastapi_endpoint_error: Optional[str] = None,
    bentoml_endpoint_error: Optional[str] = None,
) -> ConsistencyReport:
    diffs: List[ConsistencyDiff] = []
    service_errors: List[ServiceErrors] = []

    fa_sc_errors: List[str] = []
    bm_sc_errors: List[str] = []

    if fastapi_status is not None:
        sc = fastapi_status.get("last_self_check", {})
        if not sc.get("passed", False):
            fa_sc_errors.extend(sc.get("errors", []))

    if bentoml_status is not None:
        sc = bentoml_status.get("last_self_check", {})
        if not sc.get("passed", False):
            bm_sc_errors.extend(sc.get("errors", []))

    fa_endpoint_errors: List[str] = []
    bm_endpoint_errors: List[str] = []
    if fastapi_endpoint_error:
        fa_endpoint_errors.append(fastapi_endpoint_error)
    if bentoml_endpoint_error:
        bm_endpoint_errors.append(bentoml_endpoint_error)

    if fa_sc_errors or fa_endpoint_errors:
        service_errors.append(
            ServiceErrors(
                service_type="fastapi",
                self_check_errors=fa_sc_errors,
                endpoint_errors=fa_endpoint_errors,
            )
        )
    if bm_sc_errors or bm_endpoint_errors:
        service_errors.append(
            ServiceErrors(
                service_type="bentoml",
                self_check_errors=bm_sc_errors,
                endpoint_errors=bm_endpoint_errors,
            )
        )

    critical_fields = ["schema_version", "data_version", "n_features", "feature_order"]
    warning_fields = ["api_version", "model_mode"]

    if fastapi_status is not None and bentoml_status is not None:
        for field in COMPARISON_FIELDS:
            fa_val = fastapi_status.get(field)
            bm_val = bentoml_status.get(field)
            if fa_val != bm_val:
                severity = CONSISTENCY_ERROR if field in critical_fields else CONSISTENCY_WARNING
                description = (
                    f"{field}: FastAPI={_format_value(fa_val)} vs BentoML={_format_value(bm_val)}"
                )
                diffs.append(
                    ConsistencyDiff(
                        field=field,
                        fastapi_value=fa_val,
                        bentoml_value=bm_val,
                        severity=severity,
                        description=description,
                    )
                )

    has_error_diff = any(d.severity == CONSISTENCY_ERROR for d in diffs)
    has_warning_diff = any(d.severity == CONSISTENCY_WARNING for d in diffs)
    has_service_error = any(
        (se.service_type == "fastapi" and se.self_check_errors)
        or (se.service_type == "bentoml" and se.self_check_errors)
        for se in service_errors
    )
    has_endpoint_error = any(se.endpoint_errors for se in service_errors)

    if has_error_diff or has_service_error or has_endpoint_error:
        level = CONSISTENCY_ERROR
        parts = []
        if has_error_diff:
            parts.append(f"{len([d for d in diffs if d.severity == CONSISTENCY_ERROR])} critical mismatch(es)")
        if has_service_error:
            parts.append("self-check failure(s)")
        if has_endpoint_error:
            parts.append("endpoint error(s)")
        summary = "❌ Inconsistent: " + " and ".join(parts)
    elif has_warning_diff:
        level = CONSISTENCY_WARNING
        summary = f"⚠️ Partial: {len(diffs)} non-critical mismatch(es)"
    else:
        level = CONSISTENCY_OK
        summary = "✅ Consistent: FastAPI and BentoML are in sync"

    return ConsistencyReport(
        level=level,
        summary=summary,
        diffs=diffs,
        service_errors=service_errors,
    )
