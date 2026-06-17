from fastapi import APIRouter, HTTPException

from car_pricing_api.models import CarPrediction, PredictionResponse
from car_pricing_api.services.model_service import ModelService

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
