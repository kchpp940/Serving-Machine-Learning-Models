from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ModelLineage:
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    source_run_id: Optional[str] = None
    model_artifact_hash: Optional[str] = None
    candidate_metrics: Optional[Dict[str, float]] = None
    candidate_params: Optional[Dict[str, Any]] = None
    primary_metric: Optional[str] = None
    higher_is_better: Optional[bool] = None
    candidate_summary_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _find_candidate_summary() -> Optional[str]:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(base_dir, "mlflow", "candidate_summary.json"),
        os.path.join(base_dir, "candidate_summary.json"),
    ]
    env_path = os.environ.get("CANDIDATE_SUMMARY_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _find_candidate_by_hash(summary: Dict[str, Any], artifact_hash: str) -> Optional[Dict[str, Any]]:
    for c in summary.get("candidates", []):
        if c.get("model_artifact_hash") == artifact_hash:
            return c
    return None


def _find_candidate_by_name(summary: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    for c in summary.get("candidates", []):
        if c.get("model_name") == model_name:
            return c
    return None


def load_lineage(
    model_name: Optional[str] = None,
    artifact_hash: Optional[str] = None,
    candidate_summary_path: Optional[str] = None,
) -> ModelLineage:
    env_model_name = os.environ.get("MODEL_NAME")
    env_run_id = os.environ.get("SOURCE_RUN_ID")
    env_artifact_hash = os.environ.get("MODEL_ARTIFACT_HASH")

    effective_name = model_name or env_model_name
    effective_hash = artifact_hash or env_artifact_hash
    effective_run_id = env_run_id

    summary_path = candidate_summary_path or _find_candidate_summary()
    summary = None
    if summary_path and os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except (json.JSONDecodeError, IOError):
            summary = None

    candidate_info: Optional[Dict[str, Any]] = None
    if summary and effective_hash:
        candidate_info = _find_candidate_by_hash(summary, effective_hash)
    if summary and effective_name and not candidate_info:
        candidate_info = _find_candidate_by_name(summary, effective_name)

    lineage_name = effective_name
    lineage_type = None
    lineage_run_id = effective_run_id
    lineage_hash = effective_hash
    lineage_metrics = None
    lineage_params = None
    primary_metric = None
    higher_is_better = None

    if summary:
        primary_metric = summary.get("primary_metric")
        higher_is_better = summary.get("higher_is_better")
        if not lineage_name and summary.get("best_model"):
            lineage_name = summary.get("best_model")
            best_hash = summary.get("best_model_artifact_hash")
            if not lineage_hash:
                lineage_hash = best_hash
            if not candidate_info and best_hash:
                candidate_info = _find_candidate_by_hash(summary, best_hash)

    if candidate_info:
        if not lineage_name:
            lineage_name = candidate_info.get("model_name")
        if not lineage_type:
            lineage_type = candidate_info.get("model_type")
        if not lineage_run_id:
            lineage_run_id = candidate_info.get("run_id")
        if not lineage_hash:
            lineage_hash = candidate_info.get("model_artifact_hash")
        lineage_metrics = candidate_info.get("metrics")
        lineage_params = candidate_info.get("params")

    return ModelLineage(
        model_name=lineage_name,
        model_type=lineage_type,
        source_run_id=lineage_run_id,
        model_artifact_hash=lineage_hash,
        candidate_metrics=lineage_metrics,
        candidate_params=lineage_params,
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
        candidate_summary_path=summary_path,
    )
