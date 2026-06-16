from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from car_pricing.feature_schema import FeatureSchema


@dataclass
class ModelLineage:
    run_id: str = ""
    experiment_id: str = ""
    model_name: str = ""
    model_type: str = ""
    schema_version: str = ""
    data_version: str = ""
    model_artifact_hash: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    parent_run_id: Optional[str] = None
    schema: Optional[FeatureSchema] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self, include_schema: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "schema_version": self.schema_version,
            "data_version": self.data_version,
            "model_artifact_hash": self.model_artifact_hash,
            "metrics": dict(self.metrics),
            "params": dict(self.params),
            "created_at": self.created_at,
        }
        if self.parent_run_id is not None:
            result["parent_run_id"] = self.parent_run_id
        if include_schema and self.schema is not None:
            try:
                result["schema"] = self.schema.to_dict()
            except Exception:
                result["schema"] = self.schema.to_dict(include_encoders=False)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelLineage":
        schema_data = data.get("schema")
        schema = FeatureSchema.from_dict(schema_data) if schema_data else None
        return cls(
            run_id=data.get("run_id", ""),
            experiment_id=data.get("experiment_id", ""),
            model_name=data.get("model_name", ""),
            model_type=data.get("model_type", ""),
            schema_version=data.get("schema_version", ""),
            data_version=data.get("data_version", ""),
            model_artifact_hash=data.get("model_artifact_hash", ""),
            metrics=dict(data.get("metrics", {})),
            params=dict(data.get("params", {})),
            parent_run_id=data.get("parent_run_id"),
            schema=schema,
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )

    def to_runtime_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "schema_version": self.schema_version,
            "data_version": self.data_version,
            "model_artifact_hash": self.model_artifact_hash,
            "metrics": dict(self.metrics),
            "params": dict(self.params),
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "parent_run_id": self.parent_run_id,
            "created_at": self.created_at,
        }


def build_fastapi_status(lineage: ModelLineage) -> Dict[str, Any]:
    return {
        "status": "ready",
        "model_name": lineage.model_name,
        "model_type": lineage.model_type,
        "schema_version": lineage.schema_version,
        "data_version": lineage.data_version,
        "model_artifact_hash": lineage.model_artifact_hash,
        "metrics": dict(lineage.metrics),
        "loaded_at": datetime.utcnow().isoformat(),
    }
