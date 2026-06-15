from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from car_pricing.feature_schema import FeatureSchema


@dataclass
class BatchRowResult:
    row_id: Optional[str] = None
    prediction: Optional[float] = None
    error: Optional[str] = None
    field_errors: Optional[Dict[str, str]] = None


@dataclass
class BatchPredictionResponse:
    results: List[BatchRowResult] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    total_count: int = 0


def build_fallback_schema() -> dict:
    try:
        schema = FeatureSchema.default_with_encoders()
        return schema.to_api_dict()
    except Exception:
        return _minimal_fallback_schema()


def _minimal_fallback_schema() -> dict:
    from car_pricing.feature_schema import (
        FEATURE_ORDER,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        TARGET_COLUMN,
        FIELD_DISPLAY_NAMES,
    )
    return {
        "feature_order": list(FEATURE_ORDER),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "categorical_options": {},
        "display_names": dict(FIELD_DISPLAY_NAMES),
    }


def get_display_name_from_schema(field_name: str, schema: dict) -> str:
    display_names = schema.get("display_names", {})
    if field_name in display_names:
        return display_names[field_name]
    return field_name.replace("_", " ").title()


def get_default_values_from_schema(schema: dict) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    numeric_set = set(schema.get("numeric_features", []))
    categorical_set = set(schema.get("categorical_features", []))
    categorical_options = schema.get("categorical_options", {})

    for f in schema.get("feature_order", []):
        if f in numeric_set:
            defaults[f] = 0.0
        elif f in categorical_set:
            opts = categorical_options.get(f, [])
            if opts:
                defaults[f] = opts[0]["form_value"]
            else:
                defaults[f] = ""
        else:
            defaults[f] = ""
    return defaults


def validate_values_from_schema(values: Dict[str, Any], schema: dict) -> Dict[str, str]:
    errors: Dict[str, str] = {}
    numeric_set = set(schema.get("numeric_features", []))
    categorical_set = set(schema.get("categorical_features", []))
    categorical_options = schema.get("categorical_options", {})

    for f in schema.get("feature_order", []):
        if f not in values:
            errors[f] = "Missing value"
            continue
        val = values[f]
        if f in numeric_set:
            try:
                float(val)
            except (ValueError, TypeError):
                errors[f] = "Must be a number"
        elif f in categorical_set:
            opts = categorical_options.get(f, [])
            if opts:
                valid_values = {opt["form_value"] for opt in opts}
                if val not in valid_values:
                    errors[f] = f"Invalid option: {val}"
    return errors


def format_value_for_display(field_name: str, value: Any, schema: dict) -> str:
    categorical_options = schema.get("categorical_options", {})
    opts = categorical_options.get(field_name, [])
    if opts:
        for opt in opts:
            if opt["form_value"] == value:
                return opt["display"]
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value)}"
        return f"{value:.2f}"
    return str(value)
