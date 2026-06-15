import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel, Field, create_model

from car_pricing.feature_schema import FeatureSchema


def build_car_prediction_model() -> type:
    schema_dict = FeatureSchema.to_default_dict(include_encoders=False)
    feature_order = schema_dict["feature_order"]
    numeric_set = set(schema_dict["numeric_features"])
    default_values = schema_dict["default_values"]

    fields = {}
    for field_name in feature_order:
        if field_name in numeric_set:
            default_value = float(default_values[field_name])
            fields[field_name] = (float, Field(default=default_value))
        else:
            default_value = str(default_values[field_name])
            fields[field_name] = (str, Field(default=default_value))

    CarPrediction = create_model(
        "CarPrediction",
        **fields,
    )

    example = {f: default_values[f] for f in feature_order}
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
