import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, List, Optional

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
    status: str = "ok"

    class Config:
        schema_extra = {
            "example": {
                "prediction": 13295.27,
                "status": "ok",
            }
        }


class BatchPredictionRequest(BaseModel):
    rows: List[CarPrediction]

    class Config:
        schema_extra = {
            "example": {
                "rows": [
                    {
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
                    },
                    {
                        "enginesize": 150,
                        "curbweight": 2800,
                        "horsepower": 130,
                        "highwaympg": 25,
                        "carwidth": 66.0,
                        "wheelbase": 95.0,
                        "drivewheel": "fwd",
                        "citympg": 19,
                        "boreratio": 3.60,
                        "cylindernumber": "six",
                    },
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    status: str = "ok"
    count: int = 0

    class Config:
        schema_extra = {
            "example": {
                "predictions": [13295.27, 18500.50],
                "status": "ok",
                "count": 2,
            }
        }


class ServiceMetadataResponse(BaseModel):
    service_name: str = "Car Price Prediction API"
    version: str = "0.0.1"
    model_name: Optional[str] = None
    model_mode: Optional[str] = None
    n_features: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "service_name": "Car Price Prediction API",
                "version": "0.0.1",
                "model_name": "sklearn_gbr",
                "model_mode": "schema_bundle",
                "n_features": 10,
            }
        }


class ServiceStatusResponse(BaseModel):
    status: str
    uptime_seconds: Optional[float] = None
    model_loaded: bool = False

    class Config:
        schema_extra = {
            "example": {
                "status": "running",
                "uptime_seconds": 120.5,
                "model_loaded": True,
            }
        }
