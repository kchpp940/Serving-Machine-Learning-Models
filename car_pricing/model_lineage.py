from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

from car_pricing.feature_schema import FeatureSchema


@dataclass
class ModelLineage:
    run_id: str = ""
    experiment_id: str = ""
    model_name: str = "sklearn_gbr"
    model_type: str = ""
    schema_version: str = ""
    data_version: str = ""
    model_artifact_hash: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    parent_run_id: Optional[str] = None
    schema: Optional[FeatureSchema] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.schema is not None:
            data["schema"] = self.schema.to_dict()
        else:
            data.pop("schema", None)
        return data

    def to_runtime_metadata(self) -> Dict[str, Any]:
        data = self.to_dict()
        data.pop("schema", None)
        return data


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
        "n_features": lineage.params.get("n_features", 0),
    }
