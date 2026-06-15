import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List
from pydantic import BaseModel, Field

from car_pricing.feature_schema import FEATURE_ORDER


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
        json_schema_extra = {
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


class BatchPredictionRequest(BaseModel):
    items: List[CarPrediction]

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
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
                        "horsepower": 140,
                        "highwaympg": 24,
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
