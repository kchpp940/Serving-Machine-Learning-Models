import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: float
    status: str = "ok"

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 13295.27,
                "status": "ok",
            }
        }


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    count: int
    status: str = "ok"

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [13295.27, 18500.5],
                "count": 2,
                "status": "ok",
            }
        }


class ExplainResponse(BaseModel):
    prediction: float
    feature_importance: Dict[str, float]
    top_features: List[Dict[str, Any]]
    status: str = "ok"

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 13295.27,
                "feature_importance": {
                    "enginesize": 0.35,
                    "curbweight": 0.25,
                    "horsepower": 0.2,
                },
                "top_features": [
                    {"feature": "enginesize", "importance": 0.35, "value": 130},
                    {"feature": "curbweight", "importance": 0.25, "value": 2548},
                ],
                "status": "ok",
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


class ErrorResponse(BaseModel):
    status: str = "error"
    error: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "error": {
                    "code": 400,
                    "message": "无效的输入参数",
                    "error_code": "INVALID_INPUT",
                },
            }
        }
