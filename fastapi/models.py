import sys
import os
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel, Field

from car_pricing.feature_schema import FEATURE_ORDER, CATEGORICAL_FEATURES
from car_pricing.prediction_protocol import (
    PREDICTION_CURRENCY,
    ExplainResult,
    GlobalFeatureImportance as GlobalFeatureImportanceProto,
    InputFeatureValue as InputFeatureValueProto,
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
    prediction: float = Field(description=ExplainResult.PREDICTION_DESCRIPTION)
    currency: str = Field(default=PREDICTION_CURRENCY, description=ExplainResult.CURRENCY_DESCRIPTION)
    model_name: str = Field(description=ExplainResult.MODEL_NAME_DESCRIPTION)

    class Config:
        schema_extra = {
            "example": {
                "prediction": 13295.27,
                "currency": "USD",
                "model_name": "GradientBoostingRegressor",
            }
        }


class GlobalFeatureImportance(BaseModel):
    feature: str = Field(description=GlobalFeatureImportanceProto.FEATURE_DESCRIPTION)
    label: str = Field(description=GlobalFeatureImportanceProto.LABEL_DESCRIPTION)
    global_importance: float = Field(description=GlobalFeatureImportanceProto.GLOBAL_IMPORTANCE_DESCRIPTION)
    global_importance_percent: float = Field(description=GlobalFeatureImportanceProto.GLOBAL_IMPORTANCE_PERCENT_DESCRIPTION)

    class Config:
        schema_extra = {
            "example": {
                "feature": "enginesize",
                "label": "Engine Size",
                "global_importance": 0.6417,
                "global_importance_percent": 64.17,
            }
        }


class InputFeatureValue(BaseModel):
    value: Any = Field(description=InputFeatureValueProto.VALUE_DESCRIPTION)
    label: str = Field(description=InputFeatureValueProto.LABEL_DESCRIPTION)
    display: str = Field(description=InputFeatureValueProto.DISPLAY_DESCRIPTION)

    class Config:
        schema_extra = {
            "example": {
                "value": "rwd",
                "label": "Drive Wheel",
                "display": "Rear Wheel Drive (RWD)",
            }
        }


class ExplainResponse(BaseModel):
    prediction: float = Field(description=ExplainResult.PREDICTION_DESCRIPTION)
    currency: str = Field(default=PREDICTION_CURRENCY, description=ExplainResult.CURRENCY_DESCRIPTION)
    model_name: str = Field(description=ExplainResult.MODEL_NAME_DESCRIPTION)
    top_features: List[GlobalFeatureImportance] = Field(description=ExplainResult.TOP_FEATURES_DESCRIPTION)
    feature_values: Dict[str, InputFeatureValue] = Field(description=ExplainResult.FEATURE_VALUES_DESCRIPTION)

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
