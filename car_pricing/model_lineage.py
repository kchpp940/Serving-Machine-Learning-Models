from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional

from car_pricing.feature_schema import FeatureSchema


@dataclass
class ModelLineage:
    run_id: str
    experiment_id: str
    model_name: str
    model_type: str
    schema_version: str
    data_version: str
    model_artifact_hash: str
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    parent_run_id: Optional[str] = None
    schema: Optional[FeatureSchema] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        data = {
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
            data["parent_run_id"] = self.parent_run_id
        if self.schema is not None:
            data["schema"] = self.schema.to_dict()
        return data

    def to_runtime_metadata(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "schema_version": self.schema_version,
            "data_version": self.data_version,
            "model_artifact_hash": self.model_artifact_hash,
            "metrics": dict(self.metrics),
            "params": dict(self.params),
            "parent_run_id": self.parent_run_id,
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
        "params": dict(lineage.params),
    }
