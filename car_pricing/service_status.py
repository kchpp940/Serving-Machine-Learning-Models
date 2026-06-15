from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from car_pricing.feature_schema import FeatureSchema
from car_pricing.model_runtime import CarPriceModel


API_VERSION = "1.0.0"

API_BASE_SUGGESTIONS = {
    "development": "http://localhost:8000",
    "bentoml_local": "http://localhost:3000",
    "docker": "http://localhost:8080",
    "production": "https://your-api-domain.com",
}


def compute_schema_version(schema: FeatureSchema) -> str:
    schema_dict = schema.to_dict()
    for col in schema_dict.get("categorical_encoders", {}):
        schema_dict["categorical_encoders"][col]["classes"] = sorted(
            schema_dict["categorical_encoders"][col]["classes"]
        )
    payload = json.dumps(schema_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def compute_data_version(csv_path: Optional[str] = None) -> str:
    if csv_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidates = [
            os.path.join(base_dir, "Data", "cars.csv"),
            os.path.join(base_dir, "bentoml", "Data", "cars.csv"),
            os.path.join(base_dir, "mlflow", "data", "cars.csv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                csv_path = path
                break
    if csv_path is None or not os.path.exists(csv_path):
        return "unknown"
    h = hashlib.sha1()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def detect_api_base() -> str:
    env_base = os.environ.get("API_BASE_URL")
    if env_base:
        return env_base
    if os.environ.get("DOCKER_RUNTIME") or os.path.exists("/.dockerenv"):
        return API_BASE_SUGGESTIONS["docker"]
    return API_BASE_SUGGESTIONS["development"]


@dataclass
class SelfCheckResult:
    passed: bool
    checks: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_self_check(model: Optional[CarPriceModel]) -> SelfCheckResult:
    result = SelfCheckResult(passed=True, checks=[], errors=[])

    check_model_loaded = {"name": "model_loaded", "passed": False, "detail": ""}
    if model is None:
        check_model_loaded["passed"] = False
        check_model_loaded["detail"] = "CarPriceModel instance is None"
        result.passed = False
        result.errors.append("Model not loaded")
    else:
        check_model_loaded["passed"] = True
        check_model_loaded["detail"] = f"Model loaded in mode: {model.mode}"
    result.checks.append(check_model_loaded)

    if model is not None and model.schema is not None:
        check_schema = {"name": "schema_valid", "passed": False, "detail": ""}
        try:
            model.schema.validate()
            check_schema["passed"] = True
            check_schema["detail"] = "FeatureSchema validation passed"
        except Exception as e:
            check_schema["passed"] = False
            check_schema["detail"] = str(e)
            result.passed = False
            result.errors.append(f"Schema validation failed: {e}")
        result.checks.append(check_schema)

        check_dimensions = {"name": "model_dimensions_match", "passed": False, "detail": ""}
        try:
            expected = model.schema.n_features()
            actual = getattr(model.model, "n_features_in_", expected)
            if actual == expected:
                check_dimensions["passed"] = True
                check_dimensions["detail"] = f"n_features_in_={actual} matches schema features={expected}"
            else:
                check_dimensions["passed"] = False
                check_dimensions["detail"] = f"n_features_in_={actual} != schema features={expected}"
                result.passed = False
                result.errors.append(
                    f"Model dimension mismatch: n_features_in_={actual}, schema features={expected}"
                )
        except Exception as e:
            check_dimensions["passed"] = False
            check_dimensions["detail"] = str(e)
            result.passed = False
            result.errors.append(f"Dimension check error: {e}")
        result.checks.append(check_dimensions)

        check_predict = {"name": "prediction_smoke_test", "passed": False, "detail": ""}
        try:
            sample = {}
            for f in model.schema.feature_order:
                if f in model.schema.numeric_features:
                    sample[f] = 0.0
                elif f in model.schema.categorical_features:
                    classes = model.schema.categorical_classes(f)
                    sample[f] = classes[0] if classes else ""
            pred = model.predict_raw(sample)
            if len(pred) == 1 and isinstance(float(pred[0]), float):
                check_predict["passed"] = True
                check_predict["detail"] = f"Smoke prediction succeeded, output={float(pred[0]):.4f}"
            else:
                check_predict["passed"] = False
                check_predict["detail"] = f"Unexpected prediction shape: {pred}"
                result.passed = False
                result.errors.append(f"Smoke test returned unexpected output: {pred}")
        except Exception as e:
            check_predict["passed"] = False
            check_predict["detail"] = str(e)
            result.passed = False
            result.errors.append(f"Prediction smoke test failed: {e}")
        result.checks.append(check_predict)

    result.timestamp = time.time()
    return result


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
    else:
        schema_version = "unknown"
        n_features = None
        feature_order = None

    data_version = compute_data_version(data_csv_path)
    self_check_result = run_self_check(model)

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
        last_self_check=self_check_result.to_dict(),
    )
