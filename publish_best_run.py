import sys
import os
import json
import argparse

_cwd = os.getcwd()
_script_dir = os.path.abspath(os.path.dirname(__file__))

_paths_to_remove = [
    "",
    ".",
    _script_dir,
    _cwd,
]
_original_path = list(sys.path)
for p in _paths_to_remove:
    while p in sys.path:
        sys.path.remove(p)

import bentoml
import bentoml.picklable_model
import mlflow
import mlflow.sklearn

sys.path.insert(0, _script_dir)

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


def get_best_run_from_summary(summary_path=None):
    if summary_path is None:
        summary_path = os.path.join(_script_dir, "mlflow", "candidate_summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Candidate summary not found at {summary_path}. "
            "Run mlflow/train_candidates.py first."
        )

    with open(summary_path, "r") as f:
        summary = json.load(f)

    return summary


def load_model_from_run(run_id, tracking_uri, artifact_path="model"):
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def publish_best_run(summary_path=None, bento_model_name=BENTO_MODEL_NAME):
    summary = get_best_run_from_summary(summary_path)

    best_run_id = summary["best_run_id"]
    best_model_name = summary["best_model"]

    print(f"Publishing best model: {best_model_name}")
    print(f"Run ID: {best_run_id}")
    print(f"Primary metric ({summary['primary_metric']}): {summary['best_metric_value']:.4f}")

    tracking_uri = os.path.join(_script_dir, "mlflow", "mlruns")

    model_bundle = load_model_from_run(best_run_id, tracking_uri)

    car_price_model = CarPriceModel.from_sklearn_object(model_bundle)
    car_price_model.schema.validate()

    metadata = {
        "candidate_info": _sanitize_metadata({
            "best_model": best_model_name,
            "best_run_id": best_run_id,
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
            "schema_version": summary["schema_version"],
            "data_version": summary["data_version"],
            "primary_metric": summary["primary_metric"],
        },
    )

    print(f"\nModel saved to BentoML: {saved_model.tag}")
    print(f"Model path: {saved_model.path}")

    return saved_model


def main():
    parser = argparse.ArgumentParser(description="Publish best MLflow run to BentoML")
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Path to candidate_summary.json (default: mlflow/candidate_summary.json)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=BENTO_MODEL_NAME,
        help=f"BentoML model name (default: {BENTO_MODEL_NAME})",
    )

    args = parser.parse_args()

    publish_best_run(summary_path=args.summary, bento_model_name=args.model_name)


if __name__ == "__main__":
    main()
