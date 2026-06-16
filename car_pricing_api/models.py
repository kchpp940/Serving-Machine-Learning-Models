import sys
import os

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
    status: str = "ok"

    class Config:
        schema_extra = {
            "example": {
                "prediction": 13295.27,
                "status": "ok",
            }
        }


class BatchPredictionRequest(BaseModel):
    items: list[CarPrediction]

    class Config:
        schema_extra = {
            "example": {
                "items": [
                    CarPrediction.Config.schema_extra["example"],
                    {
                        "enginesize": 150,
                        "curbweight": 2700,
                        "horsepower": 130,
                        "highwaympg": 30,
                        "carwidth": 66.0,
                        "wheelbase": 95.0,
                        "drivewheel": "fwd",
                        "citympg": 24,
                        "boreratio": 3.60,
                        "cylindernumber": "four",
                    },
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    count: int
    status: str = "ok"

    class Config:
        schema_extra = {
            "example": {
                "predictions": [13295.27, 16500.00],
                "count": 2,
                "status": "ok",
            }
        }
