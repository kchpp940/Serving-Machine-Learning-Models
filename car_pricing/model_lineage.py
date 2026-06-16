from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None

from car_pricing.feature_schema import FeatureSchema
from car_pricing.versioning import (
    compute_file_hash,
    compute_dataframe_hash,
    compute_data_version,
    verify_artifact_hash,
)


@dataclass
class CandidateInfo:
    model_name: str
    model_type: str
    run_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    model_artifact_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateInfo":
        return cls(
            model_name=data["model_name"],
            model_type=data.get("model_type", ""),
            run_id=data["run_id"],
            metrics=dict(data.get("metrics", {})),
            params=dict(data.get("params", {})),
            model_artifact_hash=data.get("model_artifact_hash"),
        )


@dataclass
class CandidateSummary:
    primary_metric: str
    higher_is_better: bool
    best_model: str
    best_run_id: str
    best_metric_value: float
    schema_version: str
    data_version: str
    candidates: List[CandidateInfo] = field(default_factory=list)
    best_model_artifact_hash: Optional[str] = None
    best_model_type: Optional[str] = None
    n_candidates: Optional[int] = None
    random_state: Optional[int] = None
    test_size: Optional[float] = None

    def __post_init__(self):
        if self.n_candidates is None:
            self.n_candidates = len(self.candidates)
        if self.best_model_artifact_hash is None:
            best = self.get_best_candidate()
            if best:
                self.best_model_artifact_hash = best.model_artifact_hash
        if self.best_model_type is None:
            best = self.get_best_candidate()
            if best:
                self.best_model_type = best.model_type

    def get_best_candidate(self) -> Optional[CandidateInfo]:
        for c in self.candidates:
            if c.run_id == self.best_run_id:
                return c
        return None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["candidates"] = [c.to_dict() for c in self.candidates]
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateSummary":
        candidates = [
            CandidateInfo.from_dict(c) for c in data.get("candidates", [])
        ]
        return cls(
            primary_metric=data["primary_metric"],
            higher_is_better=data.get("higher_is_better", True),
            best_model=data["best_model"],
            best_run_id=data["best_run_id"],
            best_metric_value=data["best_metric_value"],
            schema_version=data["schema_version"],
            data_version=data["data_version"],
            candidates=candidates,
            best_model_artifact_hash=data.get("best_model_artifact_hash"),
            best_model_type=data.get("best_model_type"),
            n_candidates=data.get("n_candidates"),
            random_state=data.get("random_state"),
            test_size=data.get("test_size"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CandidateSummary":
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str) -> "CandidateSummary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_file(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


@dataclass
class ModelArtifactInfo:
    artifact_path: str
    artifact_hash: str
    artifact_size: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelArtifactInfo":
        return cls(
            artifact_path=data["artifact_path"],
            artifact_hash=data["artifact_hash"],
            artifact_size=data.get("artifact_size"),
        )


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
    candidate_summary: Optional[CandidateSummary] = None
    schema: Optional[FeatureSchema] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self, include_encoders: bool = True) -> dict:
        result = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "schema_version": self.schema_version,
            "data_version": self.data_version,
            "model_artifact_hash": self.model_artifact_hash,
            "metrics": dict(self.metrics),
            "params": dict(self.params),
            "tags": dict(self.tags),
        }
        if self.parent_run_id:
            result["parent_run_id"] = self.parent_run_id
        if self.candidate_summary:
            result["candidate_summary"] = self.candidate_summary.to_dict()
        if self.schema:
            try:
                result["schema"] = self.schema.to_dict(include_encoders=include_encoders)
            except Exception:
                result["schema"] = self.schema.to_dict(include_encoders=False)
        return result

    def to_runtime_metadata(self) -> dict:
        metadata = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "schema_version": self.schema_version,
            "data_version": self.data_version,
            "model_artifact_hash": self.model_artifact_hash,
            "metrics": dict(self.metrics),
            "params": dict(self.params),
        }
        if self.parent_run_id:
            metadata["parent_run_id"] = self.parent_run_id
        if self.schema:
            metadata["schema"] = self.schema.to_dict(include_encoders=False)
        return metadata


def _require_mlflow():
    if mlflow is None or MlflowClient is None:
        raise RuntimeError("需要安装 mlflow 才能使用 MLflow 相关功能")


class MLflowLineageReader:
    def __init__(self, tracking_uri: Optional[str] = None):
        _require_mlflow()
        self.client = MlflowClient(tracking_uri=tracking_uri)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def get_run(self, run_id: str):
        return self.client.get_run(run_id)

    def get_run_params(self, run_id: str) -> Dict[str, str]:
        run = self.client.get_run(run_id)
        return dict(run.data.params)

    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        run = self.client.get_run(run_id)
        return dict(run.data.metrics)

    def get_run_tags(self, run_id: str) -> Dict[str, str]:
        run = self.client.get_run(run_id)
        return dict(run.data.tags)

    def get_artifact_path(self, run_id: str, artifact_name: str) -> str:
        run = self.client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        if artifact_uri.startswith("file://"):
            artifact_uri = artifact_uri[7:]
        return os.path.join(artifact_uri, artifact_name)

    def list_child_runs(self, parent_run_id: str, experiment_id: Optional[str] = None):
        if experiment_id is None:
            parent_run = self.client.get_run(parent_run_id)
            experiment_id = parent_run.info.experiment_id

        filter_string = f"tags.mlflow.parentRunId = '{parent_run_id}'"
        runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
        )
        return runs

    def read_candidate_summary_from_run(self, run_id: str) -> CandidateSummary:
        params = self.get_run_params(run_id)

        if "candidate_summary" in params:
            summary_str = params["candidate_summary"]
            try:
                summary_data = json.loads(summary_str)
                return CandidateSummary.from_dict(summary_data)
            except json.JSONDecodeError:
                pass

        try:
            artifact_path = self.client.download_artifacts(
                run_id, "candidate_summary.json"
            )
            return CandidateSummary.from_file(artifact_path)
        except Exception:
            pass

        return self._rebuild_candidate_summary(run_id)

    def _rebuild_candidate_summary(self, parent_run_id: str) -> CandidateSummary:
        params = self.get_run_params(parent_run_id)
        metrics = self.get_run_metrics(parent_run_id)
        tags = self.get_run_tags(parent_run_id)

        primary_metric = params.get("primary_metric", "r2_score")
        higher_is_better = params.get("higher_is_better", "true").lower() == "true"
        best_model = params.get("best_model", tags.get("best_model", ""))
        best_run_id = params.get("best_run_id", tags.get("best_run_id", ""))
        best_metric_value = float(
            metrics.get(f"{best_model}_{primary_metric}", 0.0)
            if best_model
            else metrics.get(primary_metric, 0.0)
        )
        schema_version = params.get("schema_version", "")
        data_version = params.get("data_version", "")
        n_candidates = int(params.get("n_candidates", 0)) if params.get("n_candidates") else None
        random_state = int(params.get("random_state", 0)) if params.get("random_state") else None
        test_size = float(params.get("test_size", 0.0)) if params.get("test_size") else None

        child_runs = self.list_child_runs(parent_run_id)
        candidates = []
        best_model_type = None
        best_artifact_hash = None

        for run in child_runs:
            run_params = dict(run.data.params)
            run_metrics = dict(run.data.metrics)
            run_tags = dict(run.data.tags)

            candidate = CandidateInfo(
                model_name=run_params.get("model_name", run.info.run_name),
                model_type=run_params.get("model_type", ""),
                run_id=run.info.run_id,
                metrics=run_metrics,
                params={
                    k: v for k, v in run_params.items()
                    if k not in {"model_name", "model_type", "schema_version", "data_version", "model_artifact_hash", "n_features"}
                },
                model_artifact_hash=run_params.get(
                    "model_artifact_hash", run_tags.get("model_artifact_hash")
                ),
            )
            candidates.append(candidate)

            if run.info.run_id == best_run_id:
                best_model_type = candidate.model_type
                best_artifact_hash = candidate.model_artifact_hash

        return CandidateSummary(
            primary_metric=primary_metric,
            higher_is_better=higher_is_better,
            best_model=best_model,
            best_run_id=best_run_id,
            best_metric_value=best_metric_value,
            schema_version=schema_version,
            data_version=data_version,
            candidates=candidates,
            best_model_artifact_hash=best_artifact_hash,
            best_model_type=best_model_type,
            n_candidates=n_candidates or len(candidates),
            random_state=random_state,
            test_size=test_size,
        )

    def read_model_lineage(self, run_id: str) -> ModelLineage:
        run = self.client.get_run(run_id)
        params = dict(run.data.params)
        metrics = dict(run.data.metrics)
        tags = dict(run.data.tags)

        model_name = params.get("model_name", tags.get("mlflow.runName", ""))
        model_type = params.get("model_type", "")
        schema_version = params.get("schema_version", tags.get("schema_version", ""))
        data_version = params.get("data_version", tags.get("data_version", ""))
        model_artifact_hash = params.get(
            "model_artifact_hash", tags.get("model_artifact_hash", "")
        )
        parent_run_id = tags.get("mlflow.parentRunId")

        return ModelLineage(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            model_name=model_name,
            model_type=model_type,
            schema_version=schema_version,
            data_version=data_version,
            model_artifact_hash=model_artifact_hash,
            metrics=metrics,
            params=params,
            parent_run_id=parent_run_id,
            tags=tags,
        )

    def read_model_lineage_with_summary(self, run_id: str) -> ModelLineage:
        lineage = self.read_model_lineage(run_id)

        if lineage.parent_run_id:
            try:
                lineage.candidate_summary = self.read_candidate_summary_from_run(
                    lineage.parent_run_id
                )
            except Exception:
                pass

        return lineage


class MLflowLineageWriter:
    def __init__(self, tracking_uri: Optional[str] = None):
        _require_mlflow()
        self.client = MlflowClient(tracking_uri=tracking_uri)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def log_candidate_model_tags(
        self,
        run_id: str,
        model_name: str,
        model_type: str,
        schema: FeatureSchema,
        data_version: str,
        model_artifact_hash: str,
    ) -> None:
        schema_version = schema.schema_version()

        self.client.set_tag(run_id, "candidate_model", "true")
        self.client.set_tag(run_id, "model_name", model_name)
        self.client.set_tag(run_id, "model_type", model_type)
        self.client.set_tag(run_id, "schema_version", schema_version)
        self.client.set_tag(run_id, "data_version", data_version)
        self.client.set_tag(run_id, "model_artifact_hash", model_artifact_hash)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("schema_version", schema_version)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("model_artifact_hash", model_artifact_hash)
        mlflow.log_param("n_features", schema.n_features())

    def log_parent_run_tags(
        self,
        run_id: str,
        primary_metric: str,
        higher_is_better: bool,
        best_model: str,
        best_run_id: str,
        best_metric_value: float,
        schema_version: str,
        data_version: str,
        n_candidates: int,
        candidate_summary: CandidateSummary,
        random_state: Optional[int] = None,
        test_size: Optional[float] = None,
    ) -> None:
        self.client.set_tag(run_id, "parent_run_type", "candidate_comparison")
        self.client.set_tag(run_id, "best_model", best_model)
        self.client.set_tag(run_id, "best_run_id", best_run_id)

        mlflow.log_param("primary_metric", primary_metric)
        mlflow.log_param("higher_is_better", str(higher_is_better).lower())
        mlflow.log_param("best_model", best_model)
        mlflow.log_param("best_run_id", best_run_id)
        mlflow.log_param("best_metric_value", str(best_metric_value))
        mlflow.log_param("schema_version", schema_version)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("n_candidates", n_candidates)

        if random_state is not None:
            mlflow.log_param("random_state", random_state)
        if test_size is not None:
            mlflow.log_param("test_size", test_size)

        summary_json = candidate_summary.to_json(indent=2)
        mlflow.log_param("candidate_summary", summary_json)

        summary_path = os.path.join(
            mlflow.get_artifact_uri().replace("file://", ""),
            "candidate_summary.json",
        )
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_json)
        mlflow.log_artifact(summary_path)


def build_bentoml_metadata(lineage: ModelLineage) -> dict:
    metadata = lineage.to_runtime_metadata()
    metadata["model_source"] = "mlflow"
    return metadata


def build_fastapi_status(lineage: ModelLineage) -> dict:
    status = {
        "status": "healthy",
        "model": {
            "name": lineage.model_name,
            "type": lineage.model_type,
            "run_id": lineage.run_id,
        },
        "lineage": {
            "experiment_id": lineage.experiment_id,
            "schema_version": lineage.schema_version,
            "data_version": lineage.data_version,
            "model_artifact_hash": lineage.model_artifact_hash,
        },
        "metrics": dict(lineage.metrics),
    }
    if lineage.parent_run_id:
        status["lineage"]["parent_run_id"] = lineage.parent_run_id
    if lineage.candidate_summary:
        status["candidate_summary"] = lineage.candidate_summary.to_dict()
    return status


def create_candidate_summary_from_runs(
    runs: List,
    primary_metric: str,
    higher_is_better: bool,
    schema_version: str,
    data_version: str,
    random_state: Optional[int] = None,
    test_size: Optional[float] = None,
) -> CandidateSummary:
    candidates = []
    best_run = None
    best_value = None

    for run in runs:
        params = dict(run.data.params)
        metrics = dict(run.data.metrics)
        tags = dict(run.data.tags)

        metric_value = metrics.get(primary_metric)
        if metric_value is None:
            continue

        candidate = CandidateInfo(
            model_name=params.get("model_name", run.info.run_name),
            model_type=params.get("model_type", ""),
            run_id=run.info.run_id,
            metrics=metrics,
            params={
                k: v for k, v in params.items()
                if k not in {"model_name", "model_type", "schema_version", "data_version", "model_artifact_hash", "n_features"}
            },
            model_artifact_hash=params.get(
                "model_artifact_hash", tags.get("model_artifact_hash")
            ),
        )
        candidates.append(candidate)

        if best_value is None:
            best_value = metric_value
            best_run = candidate
        elif higher_is_better and metric_value > best_value:
            best_value = metric_value
            best_run = candidate
        elif not higher_is_better and metric_value < best_value:
            best_value = metric_value
            best_run = candidate

    if best_run is None and candidates:
        best_run = candidates[0]
        best_value = best_run.metrics.get(primary_metric, 0.0)

    return CandidateSummary(
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
        best_model=best_run.model_name if best_run else "",
        best_run_id=best_run.run_id if best_run else "",
        best_metric_value=best_value if best_value is not None else 0.0,
        schema_version=schema_version,
        data_version=data_version,
        candidates=candidates,
        best_model_artifact_hash=best_run.model_artifact_hash if best_run else None,
        best_model_type=best_run.model_type if best_run else None,
        n_candidates=len(candidates),
        random_state=random_state,
        test_size=test_size,
    )
