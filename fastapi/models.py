import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Optional, Dict, Any

from pydantic import BaseModel

from car_pricing.feature_schema import FEATURE_ORDER, CATEGORICAL_FEATURES
from car_pricing.prediction_protocol import PredictionResult as _ProtocolPredictionResult
from car_pricing.prediction_protocol import FeatureValueItem as _ProtocolFeatureValueItem
from car_pricing.prediction_protocol import TopFeatureItem as _ProtocolTopFeatureItem
from car_pricing.prediction_protocol import ExplainResult as _ProtocolExplainResult
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


# ===== 单条预测响应 Pydantic =====

class PredictionResponsePydantic(BaseModel):
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

    @classmethod
    def from_protocol(cls, proto: _ProtocolPredictionResult) -> "PredictionResponsePydantic":
        return cls(
            prediction=proto.prediction,
            currency=proto.currency,
            model_name=proto.model_name,
        )


PredictionResponse = PredictionResponsePydantic


# ===== 解释响应 Pydantic =====

class FeatureValueItemPydantic(BaseModel):
    field_name: str
    display_name: str
    raw_value: Any
    encoded_value: Optional[float] = None

    @classmethod
    def from_protocol(cls, proto: _ProtocolFeatureValueItem) -> "FeatureValueItemPydantic":
        return cls(
            field_name=proto.field_name,
            display_name=proto.display_name,
            raw_value=proto.raw_value,
            encoded_value=proto.encoded_value,
        )


class TopFeatureItemPydantic(BaseModel):
    field_name: str
    display_name: str
    global_importance: float
    global_importance_percent: float

    @classmethod
    def from_protocol(cls, proto: _ProtocolTopFeatureItem) -> "TopFeatureItemPydantic":
        return cls(
            field_name=proto.field_name,
            display_name=proto.display_name,
            global_importance=proto.global_importance,
            global_importance_percent=proto.global_importance_percent,
        )


class ExplainResponsePydantic(BaseModel):
    prediction: float
    currency: str = "USD"
    model_name: str = "sklearn_gbr"
    top_features: List[TopFeatureItemPydantic] = []
    feature_values: List[FeatureValueItemPydantic] = []

    @classmethod
    def from_protocol(cls, proto: _ProtocolExplainResult) -> "ExplainResponsePydantic":
        return cls(
            prediction=proto.prediction,
            currency=proto.currency,
            model_name=proto.model_name,
            top_features=[
                TopFeatureItemPydantic.from_protocol(x) for x in proto.top_features
            ],
            feature_values=[
                FeatureValueItemPydantic.from_protocol(x) for x in proto.feature_values
            ],
        )


# ===== 批量预测请求/响应 Pydantic =====

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
