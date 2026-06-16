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


CarPrediction = build_prediction_request_model()


def build_batch_request_model(
    item_model: Type[BaseModel],
    schema_dict: Optional[Dict[str, Any]] = None,
    model_name: str = "BatchPredictionRequest",
) -> Type[BaseModel]:
    if schema_dict is not None:
        order = schema_dict.get("feature_order", FEATURE_ORDER)
        defaults = schema_dict.get("default_values", FIELD_DEFAULT_VALUES)
    else:
        order = FEATURE_ORDER
        defaults = FIELD_DEFAULT_VALUES

    example = {
        "items": [
            {f: defaults.get(f, "") for f in order}
        ]
    }
    config = ConfigDict(json_schema_extra={"example": example})

    return create_model(
        model_name,
        items=(List[item_model], ...),
        __config__=config,
    )


BatchPredictionRequest = build_batch_request_model(CarPrediction)


def rebuild_request_models_from_schema_dict(schema_dict: Dict[str, Any]) -> None:
    import sys as _sys
    global CarPrediction, BatchPredictionRequest

    CarPrediction = build_prediction_request_model(
        schema_dict=schema_dict, model_name="CarPrediction"
    )
    BatchPredictionRequest = build_batch_request_model(
        CarPrediction, schema_dict=schema_dict, model_name="BatchPredictionRequest"
    )

    this_module = _sys.modules[__name__]
    this_module.CarPrediction = CarPrediction
    this_module.BatchPredictionRequest = BatchPredictionRequest

    schemas_module = _sys.modules.get("schemas")
    if schemas_module is not None:
        schemas_module.CarPrediction = CarPrediction
        schemas_module.BatchPredictionRequest = BatchPredictionRequest
