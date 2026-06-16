import sys
import os
from typing import Any, Dict, List, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel, Field, root_validator

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
    items: List[CarPrediction] = Field(default_factory=list)
    rows: List[CarPrediction] = Field(default_factory=list)

    @root_validator(pre=True)
    def _coerce_input(cls, values: Any) -> Dict[str, Any]:
        if isinstance(values, list):
            return {"items": values}
        if isinstance(values, dict):
            if "items" in values:
                return {"items": values["items"]}
            if "rows" in values:
                return {"items": values["rows"]}
        return values

    def normalized_items(self) -> List[CarPrediction]:
        return self.items if self.items else self.rows


class BatchPredictionResponse(BaseModel):
    predictions: List[float]

    class Config:
        schema_extra = {
            "example": {
                "predictions": [13295.27, 16500.00],
            }
        }
