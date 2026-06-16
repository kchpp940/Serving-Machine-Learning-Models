from __future__ import annotations

import os
import json
from typing import Optional, Any, Dict

from car_pricing.model_runtime import CarPriceModel
from car_pricing.model_lineage import (
    ModelLineage,
    build_fastapi_status,
    compute_file_hash,
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

    def get_lineage(self) -> ModelLineage:
        if self._lineage is None:
            model = self.model

            if os.path.exists(self._metadata_path):
                with open(self._metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                self._lineage = ModelLineage(
                    run_id=metadata.get("run_id", ""),
                    experiment_id=metadata.get("experiment_id", ""),
                    model_name=metadata.get("model_name", "sklearn_gbr"),
                    model_type=metadata.get("model_type", ""),
                    schema_version=metadata.get("schema_version", model.schema.schema_version()),
                    data_version=metadata.get("data_version", ""),
                    model_artifact_hash=metadata.get("model_artifact_hash", compute_file_hash(self._model_path)),
                    metrics=dict(metadata.get("metrics", {})),
                    params=dict(metadata.get("params", {})),
                    parent_run_id=metadata.get("parent_run_id"),
                    schema=model.schema,
                )
            else:
                model_artifact_hash = compute_file_hash(self._model_path)
                self._lineage = ModelLineage(
                    run_id="",
                    experiment_id="",
                    model_name="sklearn_gbr",
                    model_type="GradientBoostingRegressor",
                    schema_version=model.schema.schema_version(),
                    data_version="",
                    model_artifact_hash=model_artifact_hash,
                    metrics={},
                    params={"n_features": model.schema.n_features()},
                    schema=model.schema,
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
        return lineage.to_runtime_metadata()

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
