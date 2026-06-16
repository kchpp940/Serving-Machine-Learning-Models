from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

from car_pricing.feature_schema import FeatureSchema

if TYPE_CHECKING:
    from car_pricing.artifact_paths import ArtifactPaths


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

    @classmethod
    def load(cls, path: str) -> "ModelLineage":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_from_paths(cls, paths: "ArtifactPaths") -> Optional["ModelLineage"]:
        if paths.metadata_file_exists():
            return cls.load(paths.metadata_file)
        if paths.lineage_file_exists():
            return cls.load(paths.lineage_file)
        return None

    def save_metadata(self, paths: "ArtifactPaths") -> str:
        paths.ensure_model_dir()
        with open(paths.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return paths.metadata_file

    def save_status(self, paths: "ArtifactPaths") -> str:
        paths.ensure_model_dir()
        status = build_fastapi_status(self)
        with open(paths.status_file, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
        return paths.status_file

    def save_schema_snapshot(self, paths: "ArtifactPaths") -> Optional[str]:
        if self.schema is None:
            return None
        paths.ensure_model_dir()
        try:
            schema_dict = self.schema.to_dict()
        except Exception:
            schema_dict = self.schema.to_dict(include_encoders=False)
        with open(paths.schema_snapshot_file, "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, indent=2, ensure_ascii=False)
        return paths.schema_snapshot_file

    def save_lineage(self, paths: "ArtifactPaths") -> str:
        paths.ensure_model_dir()
        with open(paths.lineage_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return paths.lineage_file

    def persist(self, paths: "ArtifactPaths") -> Dict[str, str]:
        written: Dict[str, str] = {}
        written["metadata"] = self.save_metadata(paths)
        written["status"] = self.save_status(paths)
        written["lineage"] = self.save_lineage(paths)
        schema_path = self.save_schema_snapshot(paths)
        if schema_path is not None:
            written["schema_snapshot"] = schema_path
        return written


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
