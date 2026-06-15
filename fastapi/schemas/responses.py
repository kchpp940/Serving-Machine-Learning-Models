import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Generic, TypeVar, Optional, List, Dict, Any
from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    status: str = Field(..., description="Response status: 'ok' or 'error'")
    data: Optional[T] = Field(None, description="Response payload when status is 'ok'")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details when status is 'error'")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "data": {"prediction": 13295.27},
                "error": None,
            }
        }


class PredictionResponse(BaseModel):
    prediction: float

    class Config:
        json_schema_extra = {
            "example": {"prediction": 13295.27}
        }


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    count: int

    class Config:
        json_schema_extra = {
            "example": {"predictions": [13295.27, 18500.50], "count": 2}
        }


class ExplainResponse(BaseModel):
    prediction: float
    feature_importance: Dict[str, float]
    top_features: List[Dict[str, Any]]

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 13295.27,
                "feature_importance": {
                    "enginesize": 0.35,
                    "curbweight": 0.25,
                    "horsepower": 0.20,
                },
                "top_features": [
                    {"feature": "enginesize", "importance": 0.35, "value": 130},
                    {"feature": "curbweight", "importance": 0.25, "value": 2548},
                ],
            }
        }


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    model_mode: Optional[str]
    feature_count: Optional[int]
    uptime_seconds: Optional[float]

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_mode": "schema_bundle",
                "feature_count": 10,
                "uptime_seconds": 125.5,
            }
        }


class SchemaResponse(BaseModel):
    feature_order: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    target_column: str
    categorical_options: Dict[str, List[Dict[str, Any]]]

    class Config:
        json_schema_extra = {
            "example": {
                "feature_order": ["enginesize", "curbweight", "drivewheel"],
                "numeric_features": ["enginesize", "curbweight"],
                "categorical_features": ["drivewheel"],
                "target_column": "price",
                "categorical_options": {
                    "drivewheel": [
                        {"display": "Four Wheel Drive", "form_value": "4wd", "model_code": 0},
                        {"display": "Front Wheel Drive", "form_value": "fwd", "model_code": 1},
                    ]
                },
            }
        }


def success_response(data: T) -> ApiResponse[T]:
    return ApiResponse[T](status="ok", data=data, error=None)


def error_response(code: int, message: str, details: Optional[Dict[str, Any]] = None) -> ApiResponse[T]:
    error_dict = {"code": code, "message": message}
    if details:
        error_dict["details"] = details
    return ApiResponse[T](status="error", data=None, error=error_dict)
