import sys
import os
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel, Field

from car_pricing.feature_schema import FEATURE_ORDER, CATEGORICAL_FEATURES
from car_pricing.prediction_protocol import (
    PREDICTION_CURRENCY,
    GLOBAL_IMPORTANCE_DESCRIPTION,
    GLOBAL_IMPORTANCE_PERCENT_DESCRIPTION,
    EXPLAIN_TOP_FEATURES_DESCRIPTION,
    EXPLAIN_FEATURE_VALUES_DESCRIPTION,
    FIELD_PREDICTION,
    FIELD_CURRENCY,
    FIELD_MODEL_NAME,
    FIELD_TOP_FEATURES,
    FIELD_FEATURE_VALUES,
    FIELD_FEATURE,
    FIELD_LABEL,
    FIELD_VALUE,
    FIELD_DISPLAY,
    FIELD_GLOBAL_IMPORTANCE,
    FIELD_GLOBAL_IMPORTANCE_PERCENT,
    PredictionResult,
    ExplainResult,
)


class CarPrediction(BaseModel):
    enginesize: float = Field(description="Engine size in cubic centimeters", example=130)
    curbweight: float = Field(description="Curb weight of the car in pounds", example=2548)
    horsepower: float = Field(description="Horsepower output of the engine", example=111)
    highwaympg: float = Field(description="Highway miles per gallon", example=27)
    carwidth: float = Field(description="Width of the car in inches", example=64.1)
    wheelbase: float = Field(description="Wheelbase distance in inches", example=88.6)
    drivewheel: str = Field(description="Drive wheel type (fwd, rwd, 4wd)", example="rwd")
    citympg: float = Field(description="City miles per gallon", example=21)
    boreratio: float = Field(description="Engine bore ratio", example=3.47)
    cylindernumber: str = Field(description="Number of cylinders (word format)", example="four")

    class Config:
        schema_extra = {
            "example": {
                "enginesize": 130,
                "curbweight": 2548,
                "horsepower": 111,
                "highwaympg": 27,
                "carwidth": 64.1,
                "wheelbase": 88.6,
                "drivewheel": "rwd",
                "citympg": 21,
                "boreratio": 3.47,
                "cylindernumber": "four",
            }
        }


class PredictionResponse(BaseModel):
    prediction: float = Field(description="Predicted car price")
    currency: str = Field(default=PREDICTION_CURRENCY, description="Currency code of the predicted price")
    model_name: str = Field(description="Name of the ML model that produced the prediction")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 13295.27,
                "currency": "USD",
                "model_name": "GradientBoostingRegressor",
            }
        }

    @classmethod
    def from_protocol(cls, result: PredictionResult) -> "PredictionResponse":
        return cls(
            prediction=result.prediction,
            currency=result.currency,
            model_name=result.model_name,
        )


class GlobalFeatureImportance(BaseModel):
    feature: str = Field(description="Internal feature code name")
    label: str = Field(description="Human-readable feature label from the shared feature schema")
    global_importance: float = Field(description=GLOBAL_IMPORTANCE_DESCRIPTION)
    global_importance_percent: float = Field(description=GLOBAL_IMPORTANCE_PERCENT_DESCRIPTION)

    class Config:
        schema_extra = {
            "example": {
                "feature": "enginesize",
                "label": "Engine Size",
                "global_importance": 0.6417,
                "global_importance_percent": 64.17,
            }
        }

    @classmethod
    def from_protocol(cls, item) -> "GlobalFeatureImportance":
        return cls(
            feature=item.feature,
            label=item.label,
            global_importance=item.global_importance,
            global_importance_percent=item.global_importance_percent,
        )


class InputFeatureValue(BaseModel):
    value: Any = Field(description="Original input feature value")
    label: str = Field(description="Human-readable feature label from the shared feature schema")
    display: str = Field(description="Human-readable feature value representation")

    class Config:
        schema_extra = {
            "example": {
                "value": "rwd",
                "label": "Drive Wheel",
                "display": "Rear Wheel Drive (RWD)",
            }
        }

    @classmethod
    def from_protocol(cls, item) -> "InputFeatureValue":
        return cls(
            value=item.value,
            label=item.label,
            display=item.display,
        )


class ExplainResponse(BaseModel):
    prediction: float = Field(description="Predicted car price")
    currency: str = Field(default=PREDICTION_CURRENCY, description="Currency code of the predicted price")
    model_name: str = Field(description="Name of the ML model that produced the prediction")
    top_features: List[GlobalFeatureImportance] = Field(description=EXPLAIN_TOP_FEATURES_DESCRIPTION)
    feature_values: Dict[str, InputFeatureValue] = Field(description=EXPLAIN_FEATURE_VALUES_DESCRIPTION)

    class Config:
        schema_extra = {
            "example": {
                "prediction": 13295.27,
                "currency": "USD",
                "model_name": "GradientBoostingRegressor",
                "top_features": [
                    {
                        "feature": "enginesize",
                        "label": "Engine Size",
                        "global_importance": 0.6417,
                        "global_importance_percent": 64.17,
                    },
                    {
                        "feature": "curbweight",
                        "label": "Curb Weight",
                        "global_importance": 0.1659,
                        "global_importance_percent": 16.59,
                    },
                ],
                "feature_values": {
                    "enginesize": {"value": 130, "label": "Engine Size", "display": "130"},
                    "drivewheel": {"value": "rwd", "label": "Drive Wheel", "display": "Rear Wheel Drive (RWD)"},
                },
            }
        }

    @classmethod
    def from_protocol(cls, result: ExplainResult) -> "ExplainResponse":
        return cls(
            prediction=result.prediction,
            currency=result.currency,
            model_name=result.model_name,
            top_features=[
                GlobalFeatureImportance.from_protocol(item)
                for item in result.top_features
            ],
            feature_values={
                k: InputFeatureValue.from_protocol(v)
                for k, v in result.feature_values.items()
            },
        )
