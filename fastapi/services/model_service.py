import sys
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from car_pricing.model_runtime import CarPriceModel
from errors.exceptions import ModelNotFoundError


class ModelService:
    _instance: Optional["ModelService"] = None
    _model: Optional[CarPriceModel] = None
    _start_time: float = 0.0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._start_time = time.time()
        return cls._instance

    @classmethod
    def reset_for_testing(cls):
        cls._instance = None
        cls._model = None
        cls._start_time = 0.0

    def load_model(self, model_path: Optional[str] = None) -> CarPriceModel:
        if self._model is None:
            if model_path is None:
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "models",
                    "sklearn_gbr.pkl",
                )
            if not os.path.exists(model_path):
                raise ModelNotFoundError(model_path)
            self._model = CarPriceModel.from_joblib(model_path)
            self._model.schema.validate()
        return self._model

    @property
    def model(self) -> CarPriceModel:
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time


def get_model_service() -> ModelService:
    return ModelService()


def predict_single(data) -> Dict[str, Any]:
    service = get_model_service()
    predictions = service.model.predict_from_pydantic(data)
    return {
        "prediction": float(predictions[0]),
        "status": "ok",
    }


def predict_batch(items: List[Any]) -> Dict[str, Any]:
    service = get_model_service()
    predictions: List[float] = []
    for item in items:
        pred = service.model.predict_from_pydantic(item)
        predictions.append(float(pred[0]))
    return {
        "predictions": predictions,
        "count": len(predictions),
        "status": "ok",
    }


def explain_prediction(data) -> Dict[str, Any]:
    service = get_model_service()
    pred_result = predict_single(data)

    feature_importance: Dict[str, float] = {}
    raw_importance = getattr(service.model.model, "feature_importances_", None)

    if raw_importance is not None:
        for i, feature in enumerate(service.model.feature_order):
            if i < len(raw_importance):
                feature_importance[feature] = float(raw_importance[i])

    total_importance = sum(feature_importance.values()) if feature_importance else 1.0
    normalized_importance = {
        k: v / total_importance for k, v in feature_importance.items()
    }

    sorted_features = sorted(
        normalized_importance.items(), key=lambda x: x[1], reverse=True
    )

    top_features = []
    for feature, importance in sorted_features[:5]:
        top_features.append(
            {
                "feature": feature,
                "importance": round(importance, 4),
                "value": getattr(data, feature),
            }
        )

    return {
        "prediction": pred_result["prediction"],
        "feature_importance": {k: round(v, 4) for k, v in normalized_importance.items()},
        "top_features": top_features,
        "status": "ok",
    }


def get_schema_info() -> Dict[str, Any]:
    service = get_model_service()
    return {
        "feature_order": service.model.feature_order,
        "numeric_features": service.model.numeric_features,
        "categorical_features": service.model.categorical_features,
        "target_column": service.model.target_column,
        "categorical_options": {
            f: service.model.categorical_options(f)
            for f in service.model.categorical_features
        },
    }


def get_service_status() -> Dict[str, Any]:
    service = get_model_service()
    return {
        "status": "healthy" if service.is_model_loaded else "degraded",
        "model_loaded": service.is_model_loaded,
        "model_mode": service.model.mode if service.is_model_loaded else None,
        "feature_count": service.model.schema.n_features()
        if service.is_model_loaded
        else None,
        "uptime_seconds": round(service.uptime_seconds, 2),
    }
