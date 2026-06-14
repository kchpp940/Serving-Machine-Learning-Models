import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import mlflow
import bentoml
from car_pricing.model_runtime import CarPriceModel


def import_model_from_mlflow(
    run_id: str,
    model_name: str = "gbr",
    tracking_uri: str = None,
) -> dict:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)
    run_data = run.data

    params = run_data.params
    metrics = run_data.metrics
    tags = run_data.tags

    artifact_uri = run.info.artifact_uri
    print(f"Run ID: {run_id}")
    print(f"Artifact URI: {artifact_uri}")
    print(f"Params: {params}")
    print(f"Metrics: {metrics}")

    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from: {model_uri}")

    loaded_bundle = mlflow.sklearn.load_model(model_uri)

    car_model = CarPriceModel(loaded_bundle)
    print(f"Model mode: {car_model.mode}")
    print(f"Feature order: {car_model.feature_order}")
    print(f"Number of features: {car_model.schema.n_features()}")

    metadata = {
        "mlflow_run_id": run_id,
        "mlflow_artifact_uri": artifact_uri,
        "mlflow_params": params,
        "mlflow_metrics": metrics,
        "mlflow_tags": tags,
        "feature_schema": car_model.schema.to_dict(),
        "model_mode": car_model.mode,
    }

    try:
        metadata_artifact = client.download_artifacts(
            run_id, "metadata/training_metadata.json"
        )
        with open(metadata_artifact, "r", encoding="utf-8") as f:
            training_metadata = json.load(f)
        metadata["training_metadata"] = training_metadata
        print("Loaded training metadata from MLflow")
    except Exception as e:
        print(f"Could not load training metadata: {e}")

    saved_model = bentoml.sklearn.save(
        model_name,
        loaded_bundle,
        labels={
            "mlflow_run_id": run_id,
            "data_version": params.get("data_version", "unknown"),
            "model_type": "GradientBoostingRegressor",
        },
        metadata=metadata,
    )

    print(f"\nModel saved to BentoML: {saved_model.tag}")
    print(f"Model path: {saved_model.path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Import MLflow trained model to BentoML"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="MLflow run ID to import model from",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gbr",
        help="Name to use when saving to BentoML (default: gbr)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: uses current MLflow config)",
    )

    args = parser.parse_args()

    metadata = import_model_from_mlflow(
        run_id=args.run_id,
        model_name=args.model_name,
        tracking_uri=args.tracking_uri,
    )

    print("\n=== Import Complete ===")
    print(f"Data version: {metadata['mlflow_params'].get('data_version', 'N/A')}")
    print(f"Data path: {metadata['mlflow_params'].get('data_path', 'N/A')}")
    print(f"R2 Score: {metadata['mlflow_metrics'].get('R2_Score', 'N/A')}")
    print(f"Model mode: {metadata['model_mode']}")


if __name__ == "__main__":
    main()
