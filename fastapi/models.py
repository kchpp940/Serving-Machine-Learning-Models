import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel, Field, create_model

from car_pricing.feature_schema import FeatureSchema


def build_car_prediction_model() -> type:
    schema = FeatureSchema.default()
    fields = {}
    for field_name in schema.feature_order:
        if field_name in schema.numeric_features:
            default_value = float(schema.field_default_value(field_name))
            fields[field_name] = (float, Field(default=default_value))
        else:
            default_value = str(schema.field_default_value(field_name))
            fields[field_name] = (str, Field(default=default_value))

    CarPrediction = create_model(
        "CarPrediction",
        **fields,
    )

    example = {f: schema.field_default_value(f) for f in schema.feature_order}
    CarPrediction.__doc__ = "汽车价格预测输入"

    original_schema = CarPrediction.schema

    def custom_schema(*args, **kwargs):
        s = original_schema(*args, **kwargs)
        s["example"] = example
        return s

    CarPrediction.schema = classmethod(custom_schema)
    return CarPrediction


CarPrediction = build_car_prediction_model()


class PredictionResponse(BaseModel):
    prediction: float
    status: str = "ok"

    class Config:
        schema_extra = {
            "example": {
                "prediction": 13295.27,
                "status": "ok",
            }
        }
