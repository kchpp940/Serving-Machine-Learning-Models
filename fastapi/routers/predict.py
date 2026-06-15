import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import APIRouter

from schemas.requests import CarPrediction, BatchPredictionRequest
from schemas.responses import (
    ApiResponse,
    PredictionResponse,
    BatchPredictionResponse,
    ExplainResponse,
    success_response,
)
from services.model_service import predict_single, predict_batch, explain_prediction
from errors.handlers import handle_service_call

router = APIRouter(prefix="", tags=["prediction"])


@router.post("/predict", response_model=ApiResponse[PredictionResponse])
@handle_service_call
def predict(data: CarPrediction) -> ApiResponse[PredictionResponse]:
    result = predict_single(data)
    return success_response(result)


@router.post("/predict_batch", response_model=ApiResponse[BatchPredictionResponse])
@handle_service_call
def predict_batch_endpoint(data: BatchPredictionRequest) -> ApiResponse[BatchPredictionResponse]:
    result = predict_batch(data)
    return success_response(result)


@router.post("/explain", response_model=ApiResponse[ExplainResponse])
@handle_service_call
def explain(data: CarPrediction) -> ApiResponse[ExplainResponse]:
    result = explain_prediction(data)
    return success_response(result)
