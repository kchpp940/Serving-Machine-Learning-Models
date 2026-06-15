import sys
import os
import json
import argparse

_project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _project_root)

from car_pricing.safe_import import (
    clean_sys_path_for_import,
    verify_no_local_conflict,
)

_local_bentoml_dir = os.path.join(_project_root, "bentoml")

clean_sys_path_for_import("bentoml", _local_bentoml_dir)

import bentoml
import bentoml.picklable_model

verify_no_local_conflict("bentoml", _local_bentoml_dir)

clean_sys_path_for_import("mlflow")

import mlflow
import mlflow.sklearn

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from car_pricing.model_runtime import CarPriceModel


BENTO_MODEL_NAME = "car_price_model"


def _sanitize_metadata(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_metadata(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_metadata(item) for item in obj]
    elif obj is None:
        return "null"
    elif isinstance(obj, (int, float, str, bool, complex)):
        return obj
    else:
        return str(obj)


def _desanitize_value(val):
    if val == "null":
        return None
    if isinstance(val, dict):
        return {k: _desanitize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_desanitize_value(v) for v in val]
    return val


def get_candidate_summary_from_parent_run(parent_run_id: str, tracking_uri: str) -> dict:
    mlflow.set_tracking_uri(tracking_uri)

    run = mlflow.get_run(parent_run_id)

    summary_str = run.data.params.get("candidate_summary")
    if summary_str:
        try:
            return json.loads(summary_str)
        except json.JSONDecodeError:
            pass

    try:
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=parent_run_id,
            artifact_path="candidate_summary.json",
        )
        with open(artifact_path, "r") as f:
            return json.load(f)
    except Exception:
        pass

    best_run_id = run.data.params.get("best_run_id") or run.data.tags.get("best_run_id")
    best_model = run.data.params.get("best_model") or run.data.tags.get("best_model")
    primary_metric = run.data.params.get("primary_metric", "r2_score")
    higher_is_better = run.data.params.get("higher_is_better", "True") == "True"
    best_metric_value = run.data.params.get("best_metric_value")
    schema_version = run.data.params.get("schema_version")
    data_version = run.data.params.get("data_version")

    if best_run_id and best_model:
        summary = {
            "primary_metric": primary_metric,
            "higher_is_better": higher_is_better,
            "best_model": best_model,
            "best_run_id": best_run_id,
            "best_metric_value": float(best_metric_value) if best_metric_value else None,
            "schema_version": schema_version,
            "data_version": data_version,
            "candidates": [],
        }

        for child in mlflow.search_runs(
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
        ).to_dict("records"):
            candidate_run_id = child.get("run_id") or child.get("run_id")
            candidate_name = child.get("tags.candidate_model")
            if candidate_run_id and candidate_name:
                candidate_run = mlflow.get_run(candidate_run_id)
                params = {
                    k: v for k, v in candidate_run.data.params.items()
                    if k not in ["model_name", "schema_version", "data_version", "n_features"]
                }
                try:
                    params = _desanitize_value(params)
                except Exception:
                    pass

                summary["candidates"].append({
                    "model_name": candidate_name,
                    "run_id": candidate_run_id,
                    "metrics": {
                        "r2_score": float(candidate_run.data.metrics.get("r2_score", 0)),
                        "mse": float(candidate_run.data.metrics.get("mse", 0)),
                        "mae": float(candidate_run.data.metrics.get("mae", 0)),
                    },
                    "params": params,
                })

        return summary

    raise ValueError(
        f"Could not find candidate summary in parent run {parent_run_id}. "
        f"Ensure this is a valid candidate_comparison run."
    )


def get_summary_from_best_run(best_run_id: str, tracking_uri: str) -> dict:
    mlflow.set_tracking_uri(tracking_uri)

    best_run = mlflow.get_run(best_run_id)

    best_model = best_run.data.params.get("model_name") or best_run.data.tags.get("candidate_model")
    schema_version = best_run.data.params.get("schema_version")
    data_version = best_run.data.params.get("data_version")

    summary = {
        "primary_metric": "r2_score",
        "higher_is_better": True,
        "best_model": best_model or "Unknown",
        "best_run_id": best_run_id,
        "best_metric_value": float(best_run.data.metrics.get("r2_score", 0)),
        "schema_version": schema_version,
        "data_version": data_version,
        "candidates": [
            {
                "model_name": best_model or "Unknown",
                "run_id": best_run_id,
                "metrics": {
                    "r2_score": float(best_run.data.metrics.get("r2_score", 0)),
                    "mse": float(best_run.data.metrics.get("mse", 0)),
                    "mae": float(best_run.data.metrics.get("mae", 0)),
                },
                "params": _desanitize_value({
                    k: v for k, v in best_run.data.params.items()
                    if k not in ["model_name", "schema_version", "data_version", "n_features"]
                }),
            }
        ],
    }

    return summary


def find_parent_run_from_best_run(best_run_id: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    best_run = mlflow.get_run(best_run_id)

    parent_run_id = best_run.data.tags.get("mlflow.parentRunId")
    if parent_run_id:
        try:
            parent_run = mlflow.get_run(parent_run_id)
            run_type = parent_run.data.tags.get("parent_run_type")
            if run_type == "candidate_comparison":
                return parent_run_id
        except Exception:
            pass
    return None


def load_model_from_run(run_id, tracking_uri, artifact_path="model"):
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def publish_best_run(
    parent_run_id: str = None,
    best_run_id: str = None,
    tracking_uri: str = None,
    bento_model_name: str = BENTO_MODEL_NAME,
):
    if tracking_uri is None:
        tracking_uri = os.path.join(_project_root, "mlflow", "mlruns")

    if not parent_run_id and not best_run_id:
        raise ValueError("Must provide either --parent-run-id or --best-run-id")

    summary = None
    if parent_run_id:
        print(f"Loading candidate summary from parent run: {parent_run_id}")
        summary = get_candidate_summary_from_parent_run(parent_run_id, tracking_uri)
        best_run_id = summary["best_run_id"]
    else:
        print(f"Loading from best run: {best_run_id}")
        parent_run_id = find_parent_run_from_best_run(best_run_id, tracking_uri)
        if parent_run_id:
            print(f"Found parent run: {parent_run_id}")
            summary = get_candidate_summary_from_parent_run(parent_run_id, tracking_uri)
        else:
            summary = get_summary_from_best_run(best_run_id, tracking_uri)

    best_model_name = summary["best_model"]

    print(f"\n{'='*50}")
    print(f"Publishing best model: {best_model_name}")
    print(f"Run ID: {best_run_id}")
    print(f"Primary metric ({summary['primary_metric']}): {summary['best_metric_value']:.4f}")
    if parent_run_id:
        print(f"Parent run: {parent_run_id}")

    print(f"\nCandidate comparison:")
    for c in summary["candidates"]:
        marker = "  * " if c["model_name"] == best_model_name else "    "
        print(f"{marker}{c['model_name']}: {summary['primary_metric']}={c['metrics'][summary['primary_metric']]:.4f}")

    model_bundle = load_model_from_run(best_run_id, tracking_uri)

    car_price_model = CarPriceModel.from_sklearn_object(model_bundle)
    car_price_model.schema.validate()

    metadata = {
        "candidate_info": _sanitize_metadata({
            "best_model": best_model_name,
            "best_run_id": best_run_id,
            "parent_run_id": parent_run_id,
            "primary_metric": summary["primary_metric"],
            "higher_is_better": summary["higher_is_better"],
            "best_metric_value": summary["best_metric_value"],
            "schema_version": summary["schema_version"],
            "data_version": summary["data_version"],
            "candidates": summary["candidates"],
        })
    }

    saved_model = bentoml.picklable_model.save_model(
        bento_model_name,
        model_bundle,
        metadata=metadata,
        labels={
            "best_model": best_model_name,
            "best_run_id": best_run_id,
            "parent_run_id": parent_run_id or "",
            "schema_version": summary["schema_version"] or "",
            "data_version": summary["data_version"] or "",
            "primary_metric": summary["primary_metric"],
        },
    )

    print(f"\n{'='*50}")
    print(f"Model saved to BentoML: {saved_model.tag}")
    print(f"Model path: {saved_model.path}")
    print(f"\nTo start the service:")
    print(f"  cd bentoml && bentoml serve service:svc")

    return saved_model


def main():
    parser = argparse.ArgumentParser(description="Publish best MLflow run to BentoML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--parent-run-id",
        type=str,
        help="MLflow parent run ID (candidate_comparison run) containing the candidate summary",
    )
    group.add_argument(
        "--best-run-id",
        type=str,
        help="MLflow best child run ID to publish (will attempt to find parent run for full summary)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: mlflow/mlruns in project root)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=BENTO_MODEL_NAME,
        help=f"BentoML model name (default: {BENTO_MODEL_NAME})",
    )

    args = parser.parse_args()

    publish_best_run(
        parent_run_id=args.parent_run_id,
        best_run_id=args.best_run_id,
        tracking_uri=args.tracking_uri,
        bento_model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
