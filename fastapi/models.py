import sys
import os
from typing import List, Optional, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel, Field

from car_pricing.feature_schema import FEATURE_ORDER, CATEGORICAL_FEATURES


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


class FeatureContribution(BaseModel):
    rank: int = Field(description="Importance rank (1 = most important)")
    feature_name: str = Field(description="Internal feature code name")
    display_name: str = Field(description="Human-readable feature name")
    feature_value: Any = Field(description="Original input feature value")
    value_display: str = Field(description="Human-readable feature value")
    importance: float = Field(description="Feature importance score (raw)")
    importance_percent: float = Field(description="Feature importance as percentage")

    class Config:
        schema_extra = {
            "example": {
                "rank": 1,
                "feature_name": "enginesize",
                "display_name": "Engine Size",
                "feature_value": 130,
                "value_display": "130",
                "importance": 0.6417,
                "importance_percent": 64.17,
            }
        }


class PredictionResponse(BaseModel):
    prediction: float = Field(description="Predicted car price")
    status: str = Field(default="ok", description="Response status")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 13295.27,
                "status": "ok",
            }
        }


class PredictionWithExplanationResponse(PredictionResponse):
    model_name: str = Field(description="Name of the ML model used")
    top_features: List[FeatureContribution] = Field(description="Top N most important features for this prediction")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 13295.27,
                "status": "ok",
                "model_name": "GradientBoostingRegressor",
                "top_features": [
                    {
                        "rank": 1,
                        "feature_name": "enginesize",
                        "display_name": "Engine Size",
                        "feature_value": 130,
                        "value_display": "130",
                        "importance": 0.6417,
                        "importance_percent": 64.17,
                    },
                    {
                        "rank": 2,
                        "feature_name": "curbweight",
                        "display_name": "Curb Weight",
                        "feature_value": 2548,
                        "value_display": "2548",
                        "importance": 0.1659,
                        "importance_percent": 16.59,
                    },
                ],
            }
        }


class ExplainResponse(PredictionWithExplanationResponse):
    all_features: List[FeatureContribution] = Field(description="All features sorted by importance")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 13295.27,
                "status": "ok",
                "model_name": "GradientBoostingRegressor",
                "top_features": [
                    {
                        "rank": 1,
                        "feature_name": "enginesize",
                        "display_name": "Engine Size",
                        "feature_value": 130,
                        "value_display": "130",
                        "importance": 0.6417,
                        "importance_percent": 64.17,
                    },
                ],
                "all_features": [
                    {
                        "rank": 1,
                        "feature_name": "enginesize",
                        "display_name": "Engine Size",
                        "feature_value": 130,
                        "value_display": "130",
                        "importance": 0.6417,
                        "importance_percent": 64.17,
                    },
                ],
            }
        }
