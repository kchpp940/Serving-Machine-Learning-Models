from __future__ import annotations

import os
import json
from typing import Optional, Any, Dict

from car_pricing.model_runtime import CarPriceModel
from car_pricing.model_lineage import (
    load_runtime_lineage,
    build_fastapi_status,
    build_runtime_metadata,
)


class ModelService:
    _instance: Optional["ModelService"] = None

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")

        self._model_dir = os.path.abspath(model_dir)
        self._model_path = os.path.join(self._model_dir, "sklearn_gbr.pkl")
        self._metadata_path = os.path.join(self._model_dir, "model_metadata.json")

        self._model: Optional[CarPriceModel] = None
        self._lineage: Optional[ModelLineage] = None

    @classmethod
    def get_instance(cls, model_dir: Optional[str] = None) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls(model_dir=model_dir)
        return cls._instance

    def load(self) -> None:
        if self._model is None:
            if not os.path.exists(self._model_path):
                raise RuntimeError(f"模型文件不存在: {self._model_path}")
            self._model = CarPriceModel.from_joblib(self._model_path)
            self._model.schema.validate()

    @property
    def model(self) -> CarPriceModel:
        if self._model is None:
            self.load()
        return self._model

    def get_lineage(self):
        if self._lineage is None:
            self._lineage = load_runtime_lineage(
                model_path=self._model_path,
                metadata_path=self._metadata_path,
            )
        return self._lineage

    def predict(self, data) -> float:
        model = self.model
        predictions = model.predict_from_pydantic(data)
        return float(predictions[0])

    def get_schema(self) -> Dict[str, Any]:
        model = self.model
        return {
            "feature_order": model.feature_order,
            "numeric_features": model.numeric_features,
            "categorical_features": model.categorical_features,
            "target_column": model.target_column,
            "categorical_options": {
                f: model.categorical_options(f) for f in model.categorical_features
            },
        }

    def get_status(self) -> Dict[str, Any]:
        lineage = self.get_lineage()
        return build_fastapi_status(lineage)

    def get_metadata(self) -> Dict[str, Any]:
        lineage = self.get_lineage()
        return build_runtime_metadata(lineage)

    def get_health(self) -> Dict[str, Any]:
        try:
            model = self.model
            lineage = self.get_lineage()
            return {
                "status": "healthy",
                "model_loaded": True,
                "model_name": lineage.model_name,
                "model_type": lineage.model_type,
                "schema_version": lineage.schema_version,
                "data_version": lineage.data_version,
                "n_features": model.schema.n_features(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(e),
            }
