import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from car_pricing.feature_schema import FEATURE_ORDER, CATEGORICAL_FEATURES
from car_pricing.prediction_protocol import (
    DEFAULT_CURRENCY,
    DEFAULT_MODEL_NAME,
    DEFAULT_STATUS,
    SINGLE_PREDICTION_FIELDS,
    BATCH_ITEM_FIELDS,
    BATCH_RESPONSE_FIELDS,
)


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
    currency: str = DEFAULT_CURRENCY
    model_name: str = DEFAULT_MODEL_NAME

    class Config:
        schema_extra = {
            "example": {
                "prediction": 13295.27,
                "currency": DEFAULT_CURRENCY,
                "model_name": "sklearn_gbr",
            }
        }


class BatchPredictionRequest(BaseModel):
    records: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "records": [
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
                        "enginesize": 152,
                        "curbweight": 3086,
                        "horsepower": 154,
                        "highwaympg": 26,
                        "carwidth": 66.3,
                        "wheelbase": 99.8,
                        "drivewheel": "fwd",
                        "citympg": 19,
                        "boreratio": 3.54,
                        "cylindernumber": "six",
                    },
                ]
            }
        }


class BatchPredictionItem(BaseModel):
    row_index: int
    prediction: Optional[float] = None
    currency: str = DEFAULT_CURRENCY
    model_name: str = DEFAULT_MODEL_NAME
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    status: str = DEFAULT_STATUS
    total_records: int
    valid_count: int
    invalid_count: int
    results: List[BatchPredictionItem]

    class Config:
        schema_extra = {
            "example": {
                "status": DEFAULT_STATUS,
                "total_records": 2,
                "valid_count": 2,
                "invalid_count": 0,
                "results": [
                    {
                        "row_index": 0,
                        "prediction": 13295.27,
                        "currency": DEFAULT_CURRENCY,
                        "model_name": "sklearn_gbr",
                        "error": None,
                    },
                    {
                        "row_index": 1,
                        "prediction": 18945.63,
                        "currency": DEFAULT_CURRENCY,
                        "model_name": "sklearn_gbr",
                        "error": None,
                    },
                ],
            }
        }
