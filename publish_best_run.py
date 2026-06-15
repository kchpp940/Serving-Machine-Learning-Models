import sys
import os
import json
import argparse
import hashlib
import tempfile

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

        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        experiment_ids = [exp.experiment_id for exp in experiments]

        child_runs = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
        )

        for child in _runs_to_dicts(child_runs):
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


def _runs_to_dicts(runs) -> list:
    result = []
    for run in runs:
        run_dict = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
        }
        for tag_key, tag_value in run.data.tags.items():
            run_dict[f"tags.{tag_key}"] = tag_value
        result.append(run_dict)
    return result


def get_child_runs_from_parent(parent_run_id: str, tracking_uri: str) -> list:
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    experiment_ids = [exp.experiment_id for exp in experiments]

    child_runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )

    return _runs_to_dicts(child_runs)


def rebuild_candidate_summary_from_child_runs(
    parent_run_id: str,
    tracking_uri: str,
    primary_metric: str,
    higher_is_better: bool,
) -> dict:
    mlflow.set_tracking_uri(tracking_uri)

    child_runs = get_child_runs_from_parent(parent_run_id, tracking_uri)

    candidates = []
    for child in child_runs:
        run_id = child.get("run_id")
        if not run_id:
            continue

        run = mlflow.get_run(run_id)
        model_name = run.data.params.get("model_name") or run.data.tags.get("candidate_model")

        if not model_name:
            continue

        params = {
            k: v for k, v in run.data.params.items()
            if k not in ["model_name", "schema_version", "data_version", "n_features"]
        }

        try:
            params = _desanitize_value(params)
        except Exception:
            pass

        metrics = {
            metric: float(run.data.metrics.get(metric, 0))
            for metric in ["r2_score", "mse", "mae"]
            if metric in run.data.metrics
        }

        for metric_key, metric_value in run.data.metrics.items():
            if metric_key not in metrics:
                try:
                    metrics[metric_key] = float(metric_value)
                except (ValueError, TypeError):
                    metrics[metric_key] = metric_value

        candidate = {
            "model_name": model_name,
            "run_id": run_id,
            "metrics": metrics,
            "params": params,
            "schema_version": run.data.params.get("schema_version"),
            "data_version": run.data.params.get("data_version"),
        }

        candidates.append(candidate)

    def metric_value(c):
        return c.get("metrics", {}).get(primary_metric, -float("inf"))

    candidates_sorted = sorted(
        candidates,
        key=metric_value,
        reverse=higher_is_better,
    )

    best = candidates_sorted[0] if candidates_sorted else None

    schema_versions = set(c.get("schema_version") for c in candidates if c.get("schema_version"))
    data_versions = set(c.get("data_version") for c in candidates if c.get("data_version"))

    summary = {
        "primary_metric": primary_metric,
        "higher_is_better": higher_is_better,
        "best_model": best.get("model_name") if best else None,
        "best_run_id": best.get("run_id") if best else None,
        "best_metric_value": best.get("metrics", {}).get(primary_metric) if best else None,
        "schema_version": schema_versions.pop() if len(schema_versions) == 1 else None,
        "data_version": data_versions.pop() if len(data_versions) == 1 else None,
        "candidates": candidates_sorted,
        "schema_versions": list(schema_versions),
        "data_versions": list(data_versions),
    }

    return summary


def get_model_artifact_hash(run_id: str, tracking_uri: str, artifact_path: str = "model") -> str:
    mlflow.set_tracking_uri(tracking_uri)

    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        )
    except Exception:
        return None

    if not local_path or not os.path.exists(local_path):
        return None

    hasher = hashlib.sha256()

    if os.path.isdir(local_path):
        for root, dirs, files in os.walk(local_path):
            dirs.sort()
            for filename in sorted(files):
                filepath = os.path.join(root, filename)
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
    else:
        with open(local_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

    return hasher.hexdigest()


def verify_consistency(
    parent_summary: dict,
    rebuilt_summary: dict,
    tracking_uri: str,
) -> dict:
    result = {
        "all_checks_passed": True,
        "checks": {},
        "mismatches": [],
    }

    checks = [
        ("best_run_id", parent_summary.get("best_run_id"), rebuilt_summary.get("best_run_id")),
        ("best_model", parent_summary.get("best_model"), rebuilt_summary.get("best_model")),
        ("primary_metric", parent_summary.get("primary_metric"), rebuilt_summary.get("primary_metric")),
        ("higher_is_better", parent_summary.get("higher_is_better"), rebuilt_summary.get("higher_is_better")),
        ("schema_version", parent_summary.get("schema_version"), rebuilt_summary.get("schema_version")),
        ("data_version", parent_summary.get("data_version"), rebuilt_summary.get("data_version")),
    ]

    for check_name, parent_val, rebuilt_val in checks:
        passed = parent_val == rebuilt_val
        result["checks"][check_name] = {
            "passed": passed,
            "parent_value": parent_val,
            "rebuilt_value": rebuilt_val,
        }
        if not passed:
            result["all_checks_passed"] = False
            result["mismatches"].append(
                f"{check_name}: parent={parent_val}, rebuilt={rebuilt_val}"
            )

    parent_best_metric = parent_summary.get("best_metric_value")
    rebuilt_best_metric = rebuilt_summary.get("best_metric_value")
    metric_match = False
    if parent_best_metric is not None and rebuilt_best_metric is not None:
        try:
            metric_match = abs(float(parent_best_metric) - float(rebuilt_best_metric)) < 1e-9
        except (ValueError, TypeError):
            metric_match = parent_best_metric == rebuilt_best_metric

    result["checks"]["best_metric_value"] = {
        "passed": metric_match,
        "parent_value": parent_best_metric,
        "rebuilt_value": rebuilt_best_metric,
    }
    if not metric_match:
        result["all_checks_passed"] = False
        result["mismatches"].append(
            f"best_metric_value: parent={parent_best_metric}, rebuilt={rebuilt_best_metric}"
        )

    best_run_id = rebuilt_summary.get("best_run_id")
    model_artifact_hash = None
    if best_run_id:
        model_artifact_hash = get_model_artifact_hash(best_run_id, tracking_uri)

    result["checks"]["model_artifact_hash"] = {
        "passed": model_artifact_hash is not None,
        "rebuilt_value": model_artifact_hash,
    }
    if model_artifact_hash is None:
        result["all_checks_passed"] = False
        result["mismatches"].append("model_artifact_hash: could not download or hash model artifact")

    parent_candidates = parent_summary.get("candidates", [])
    rebuilt_candidates = rebuilt_summary.get("candidates", [])
    candidate_count_match = len(parent_candidates) == len(rebuilt_candidates)

    result["checks"]["candidate_count"] = {
        "passed": candidate_count_match,
        "parent_value": len(parent_candidates),
        "rebuilt_value": len(rebuilt_candidates),
    }
    if not candidate_count_match:
        result["all_checks_passed"] = False
        result["mismatches"].append(
            f"candidate_count: parent={len(parent_candidates)}, rebuilt={len(rebuilt_candidates)}"
        )

    primary_metric = parent_summary.get("primary_metric", "r2_score")
    parent_candidate_metrics = {}
    for c in parent_candidates:
        cid = c.get("model_name") or c.get("run_id")
        if cid:
            parent_candidate_metrics[cid] = c.get("metrics", {}).get(primary_metric)

    rebuilt_candidate_metrics = {}
    for c in rebuilt_candidates:
        cid = c.get("model_name") or c.get("run_id")
        if cid:
            rebuilt_candidate_metrics[cid] = c.get("metrics", {}).get(primary_metric)

    all_candidate_metrics_match = True
    for cid in parent_candidate_metrics:
        if cid in rebuilt_candidate_metrics:
            pm = parent_candidate_metrics[cid]
            rm = rebuilt_candidate_metrics[cid]
            if pm is not None and rm is not None:
                try:
                    if abs(float(pm) - float(rm)) >= 1e-9:
                        all_candidate_metrics_match = False
                        break
                except (ValueError, TypeError):
                    if pm != rm:
                        all_candidate_metrics_match = False
                        break
            elif pm != rm:
                all_candidate_metrics_match = False
                break
        else:
            all_candidate_metrics_match = False
            break

    result["checks"]["all_candidate_metrics"] = {
        "passed": all_candidate_metrics_match,
        "parent_values": parent_candidate_metrics,
        "rebuilt_values": rebuilt_candidate_metrics,
    }
    if not all_candidate_metrics_match:
        result["all_checks_passed"] = False
        result["mismatches"].append(
            "all_candidate_metrics: some candidate metrics differ between parent summary and child runs"
        )

    return result, model_artifact_hash


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
    primary_metric = summary["primary_metric"]
    higher_is_better = summary["higher_is_better"]

    print(f"\n{'='*50}")
    print(f"Publishing best model: {best_model_name}")
    print(f"Run ID: {best_run_id}")
    print(f"Primary metric ({primary_metric}): {summary['best_metric_value']:.4f}")
    if parent_run_id:
        print(f"Parent run: {parent_run_id}")

    print(f"\nCandidate comparison (from parent summary):")
    for c in summary["candidates"]:
        marker = "  * " if c["model_name"] == best_model_name else "    "
        print(f"{marker}{c['model_name']}: {primary_metric}={c['metrics'][primary_metric]:.4f}")

    consistency_result = None
    model_artifact_hash = None

    if parent_run_id:
        print(f"\n{'='*50}")
        print("Verifying consistency: rebuilding summary from child runs...")

        rebuilt_summary = rebuild_candidate_summary_from_child_runs(
            parent_run_id=parent_run_id,
            tracking_uri=tracking_uri,
            primary_metric=primary_metric,
            higher_is_better=higher_is_better,
        )

        print(f"\nRebuilt candidate comparison (from child runs):")
        for c in rebuilt_summary["candidates"]:
            marker = "  * " if c.get("is_best") or c["model_name"] == rebuilt_summary.get("best_model") else "    "
            metric_val = c.get("metrics", {}).get(primary_metric, 0)
            try:
                metric_str = f"{float(metric_val):.4f}"
            except (ValueError, TypeError):
                metric_str = str(metric_val)
            print(f"{marker}{c['model_name']}: {primary_metric}={metric_str}")

        print(f"\n{'='*50}")
        print("Running consistency checks...")

        consistency_result, model_artifact_hash = verify_consistency(
            parent_summary=summary,
            rebuilt_summary=rebuilt_summary,
            tracking_uri=tracking_uri,
        )

        print(f"\nConsistency check results:")
        for check_name, check_data in consistency_result["checks"].items():
            status = "✓ PASS" if check_data["passed"] else "✗ FAIL"
            print(f"  {status}: {check_name}")

        if not consistency_result["all_checks_passed"]:
            print(f"\n{'='*50}")
            print("ERROR: Consistency checks failed!")
            print("Mismatches:")
            for mismatch in consistency_result["mismatches"]:
                print(f"  - {mismatch}")
            print("\nThe parent run summary does not match the actual child run data.")
            print("This may indicate data tampering or corruption.")
            print("Aborting publish.")
            raise ValueError(
                "Consistency checks failed. "
                f"Mismatches: {'; '.join(consistency_result['mismatches'])}"
            )

        print(f"\n✓ All consistency checks passed!")
        if model_artifact_hash:
            print(f"  Model artifact hash: {model_artifact_hash[:16]}...")

    model_bundle = load_model_from_run(best_run_id, tracking_uri)

    car_price_model = CarPriceModel.from_sklearn_object(model_bundle)
    car_price_model.schema.validate()

    metadata = {
        "candidate_info": _sanitize_metadata({
            "best_model": best_model_name,
            "best_run_id": best_run_id,
            "parent_run_id": parent_run_id,
            "primary_metric": primary_metric,
            "higher_is_better": higher_is_better,
            "best_metric_value": summary["best_metric_value"],
            "schema_version": summary["schema_version"],
            "data_version": summary["data_version"],
            "candidates": summary["candidates"],
        }),
        "consistency_check": _sanitize_metadata({
            "performed": parent_run_id is not None,
            "all_checks_passed": consistency_result["all_checks_passed"] if consistency_result else None,
            "model_artifact_hash": model_artifact_hash,
            "checks": consistency_result["checks"] if consistency_result else None,
            "verified_at": os.popen('date -u +"%Y-%m-%dT%H:%M:%SZ"').read().strip(),
        }) if consistency_result else _sanitize_metadata({
            "performed": False,
            "reason": "No parent run found - single model publish",
        }),
    }

    labels = {
        "best_model": best_model_name,
        "best_run_id": best_run_id,
        "parent_run_id": parent_run_id or "",
        "schema_version": summary["schema_version"] or "",
        "data_version": summary["data_version"] or "",
        "primary_metric": primary_metric,
    }

    if consistency_result:
        labels["consistency_verified"] = "true" if consistency_result["all_checks_passed"] else "false"
        if model_artifact_hash:
            labels["artifact_hash"] = model_artifact_hash[:16]

    saved_model = bentoml.picklable_model.save_model(
        bento_model_name,
        model_bundle,
        metadata=metadata,
        labels=labels,
    )

    print(f"\n{'='*50}")
    print(f"Model saved to BentoML: {saved_model.tag}")
    print(f"Model path: {saved_model.path}")
    if consistency_result:
        print(f"Consistency verified: {'Yes' if consistency_result['all_checks_passed'] else 'No'}")
        if model_artifact_hash:
            print(f"Model artifact hash: {model_artifact_hash}")
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
