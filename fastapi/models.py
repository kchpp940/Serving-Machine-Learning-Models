import sys
import os
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel, Field

from car_pricing.feature_schema import FEATURE_ORDER, CATEGORICAL_FEATURES


class CarPrediction(BaseModel):
    enginesize: float
    curbweight: float
    horsepower: float
    highwaympg: float
    carwidth: float
    wheelbase: float
    drivewheel: str
    citympg: float
    boreratio: float
    cylindernumber: str

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
    prediction: float
    currency: str = "USD"
    model_name: str = "sklearn_gbr"

    class Config:
        schema_extra = {
            "example": {
                "prediction": 13295.27,
                "currency": "USD",
                "model_name": "sklearn_gbr",
            }
        }


class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    detail: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "error": "InvalidInput",
                "detail": "enginesize must be a positive number",
            }
        }


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "car-price-prediction-api"
    version: str
    model_loaded: bool

    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "service": "car-price-prediction-api",
                "version": "0.0.1",
                "model_loaded": True,
            }
        }
