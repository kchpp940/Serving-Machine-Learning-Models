import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Optional, Dict, Any

from pydantic import BaseModel

from car_pricing.feature_schema import FEATURE_ORDER, CATEGORICAL_FEATURES
from car_pricing.prediction_protocol import BatchRowResult as _ProtocolBatchRowResult
from car_pricing.prediction_protocol import BatchPredictionResponse as _ProtocolBatchPredictionResponse


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
    rows: List[Dict[str, Any]]
    row_ids: Optional[List[str]] = None


class BatchRowResultPydantic(BaseModel):
    row_id: Optional[str] = None
    prediction: Optional[float] = None
    error: Optional[str] = None
    field_errors: Optional[Dict[str, str]] = None

    @classmethod
    def from_protocol(cls, proto: _ProtocolBatchRowResult) -> "BatchRowResultPydantic":
        return cls(
            row_id=proto.row_id,
            prediction=proto.prediction,
            error=proto.error,
            field_errors=proto.field_errors,
        )


class BatchPredictionResponsePydantic(BaseModel):
    results: List[BatchRowResultPydantic]
    success_count: int
    error_count: int
    total_count: int

    @classmethod
    def from_protocol(cls, proto: _ProtocolBatchPredictionResponse) -> "BatchPredictionResponsePydantic":
        return cls(
            results=[BatchRowResultPydantic.from_protocol(r) for r in proto.results],
            success_count=proto.success_count,
            error_count=proto.error_count,
            total_count=proto.total_count,
        )
