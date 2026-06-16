import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, ConfigDict, create_model

from car_pricing.feature_schema import (
    FeatureSchema,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FIELD_DEFAULT_VALUES,
)


def _build_fields_from_schema(schema: FeatureSchema) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for field_name in schema.feature_order:
        if field_name in schema.numeric_features:
            default = schema.field_default_value(field_name)
            fields[field_name] = (float, default)
        elif field_name in schema.categorical_features:
            default = schema.field_default_value(field_name)
            fields[field_name] = (str, default)
    return fields


def build_prediction_request_model(
    schema: Optional[FeatureSchema] = None,
    schema_dict: Optional[Dict[str, Any]] = None,
    model_name: str = "CarPrediction",
) -> Type[BaseModel]:
    if schema_dict is not None:
        schema = FeatureSchema.from_dict(schema_dict)
    elif schema is None:
        schema = FeatureSchema.default()

    fields = _build_fields_from_schema(schema)
    example = {f: schema.field_default_value(f) for f in schema.feature_order}
    config = ConfigDict(json_schema_extra={"example": example})

    model_cls = create_model(model_name, __config__=config, **fields)
    return model_cls


def build_batch_request_model(
    item_model: Type[BaseModel],
    schema_dict: Optional[Dict[str, Any]] = None,
    model_name: str = "BatchPredictionRequest",
    field_name: str = "rows",
) -> Type[BaseModel]:
    if schema_dict is not None:
        order = schema_dict.get("feature_order", FEATURE_ORDER)
        defaults = schema_dict.get("default_values", FIELD_DEFAULT_VALUES)
    else:
        order = FEATURE_ORDER
        defaults = FIELD_DEFAULT_VALUES

    example = {
        field_name: [
            {f: defaults.get(f, "") for f in order}
        ]
    }
    config = ConfigDict(json_schema_extra={"example": example})

    return create_model(
        model_name,
        **{field_name: (List[item_model], ...)},
        __config__=config,
    )
