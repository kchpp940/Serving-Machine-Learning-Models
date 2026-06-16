import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import APIRouter

from schemas.requests import CarPrediction, BatchPredictionRequest
from schemas.responses import (
    PredictionResponse,
    BatchPredictionResponse,
    ExplainResponse,
    ErrorResponse,
)
from services.model_service import predict_single, predict_batch, explain_prediction

router = APIRouter(prefix="", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def predict(data: CarPrediction):
    return predict_single(data)


@router.post("/predict_batch", response_model=BatchPredictionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def predict_batch_endpoint(data: BatchPredictionRequest):
    return predict_batch(data.items)


@router.post("/explain", response_model=ExplainResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def explain(data: CarPrediction):
    return explain_prediction(data)
