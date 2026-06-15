from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

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
