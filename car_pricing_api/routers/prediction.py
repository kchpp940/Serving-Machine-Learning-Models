import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import APIRouter, HTTPException

from models import (
    CarPrediction,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from services.model_service import ModelService

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(data: CarPrediction):
    try:
        service = ModelService.get_instance()
        value = service.predict(data)
        return PredictionResponse(prediction=value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@router.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(data: BatchPredictionRequest):
    try:
        service = ModelService.get_instance()
        items = data.normalized_items()
        if not items:
            raise ValueError("Empty batch: no items provided.")
        values = service.predict_batch(items)
        return BatchPredictionResponse(predictions=values)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")
