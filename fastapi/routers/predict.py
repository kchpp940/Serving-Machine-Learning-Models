import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Any, Dict, List, Type, Union, Annotated
from fastapi import APIRouter, Body
from pydantic import BaseModel, model_validator

from schemas import (
    build_prediction_request_model,
    build_batch_request_model,
    PredictionResponse,
    BatchPredictionResponse,
    ExplainResponse,
)
from services import (
    predict_single,
    predict_batch,
    explain_prediction,
)


def build_predict_router(schema_dict: Dict[str, Any]) -> APIRouter:
    CarPrediction = build_prediction_request_model(
        schema_dict=schema_dict, model_name="CarPrediction"
    )

    BatchWithRows = build_batch_request_model(
        CarPrediction, schema_dict=schema_dict,
        model_name="BatchWithRows", field_name="rows"
    )
    BatchWithItems = build_batch_request_model(
        CarPrediction, schema_dict=schema_dict,
        model_name="BatchWithItems", field_name="items"
    )

    ListOfPredictions = List[CarPrediction]

    router = APIRouter(prefix="", tags=["prediction"])

    @router.post("/predict", response_model=PredictionResponse)
    def predict(data: CarPrediction) -> PredictionResponse:
        return predict_single(data)

    @router.post("/predict_batch", response_model=BatchPredictionResponse)
    def predict_batch_endpoint(
        body: Annotated[
            Union[ListOfPredictions, BatchWithRows, BatchWithItems],
            Body(
                examples=[
                    {"rows": [
                        {f: schema_dict.get("default_values", {}).get(f, 0.0)
                         for f in schema_dict.get("feature_order", [])}
                    ]},
                    [
                        {f: schema_dict.get("default_values", {}).get(f, 0.0)
                         for f in schema_dict.get("feature_order", [])}
                    ],
                ],
            ),
        ],
    ) -> BatchPredictionResponse:
        if isinstance(body, list):
            rows = body
        elif isinstance(body, BatchWithRows):
            rows = body.rows
        elif isinstance(body, BatchWithItems):
            rows = body.items
        else:
            raise ValueError(f"Unsupported batch request format: {type(body).__name__}")
        return predict_batch(rows)

    @router.post("/explain", response_model=ExplainResponse)
    def explain(data: CarPrediction) -> ExplainResponse:
        return explain_prediction(data)

    return router
