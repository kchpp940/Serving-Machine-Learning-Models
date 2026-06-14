import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
import mlflow
import bentoml

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import calculate_schema_version


def import_from_mlflow(
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
    print(f"=" * 60)
    print(f"Importing model from MLflow")
    print(f"=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Artifact URI: {artifact_uri}")
    print(f"Data version: {params.get('data_version', 'N/A')}")
    print(f"R2 Score: {metrics.get('R2_Score', 'N/A')}")

    model_uri = f"runs:/{run_id}/model"
    print(f"\nLoading model bundle from: {model_uri}")
    loaded_bundle = mlflow.sklearn.load_model(model_uri)

    car_model = CarPriceModel(loaded_bundle)
    print(f"Model mode: {car_model.mode}")
    print(f"Schema version: {car_model.schema_version}")
    print(f"Feature order: {car_model.feature_order}")
    print(f"Number of features: {car_model.schema.n_features()}")

    expected_schema_version = params.get("schema_version")
    if expected_schema_version and car_model.schema_version != expected_schema_version:
        raise ValueError(
            f"Schema version mismatch! MLflow run recorded {expected_schema_version}, "
            f"but current code produces {car_model.schema_version}. "
            f"Feature processing logic may have drifted."
        )
    print(f"✓ Schema version verified: {car_model.schema_version}")

    training_metadata = None
    try:
        metadata_artifact = client.download_artifacts(
            run_id, "metadata/training_metadata.json"
        )
        with open(metadata_artifact, "r", encoding="utf-8") as f:
            training_metadata = json.load(f)
        print("✓ Loaded training metadata from MLflow")

        if training_metadata.get("schema_version") != car_model.schema_version:
            raise ValueError(
                f"Schema version in training_metadata does not match!"
            )
    except Exception as e:
        print(f"Note: Could not load full training metadata: {e}")

    metadata = {
        "mlflow_run_id": run_id,
        "mlflow_artifact_uri": artifact_uri,
        "mlflow_params": params,
        "mlflow_metrics": metrics,
        "mlflow_tags": tags,
        "feature_schema": car_model.schema.to_dict(),
        "model_mode": car_model.mode,
        "schema_version": car_model.schema_version,
        "data_version": params.get("data_version"),
        "data_path": params.get("data_path"),
        "bento_source": "mlflow_import",
        "import_timestamp": pd.Timestamp.now().isoformat(),
    }

    if training_metadata:
        metadata["training_metadata"] = training_metadata

    current_schema_version = calculate_schema_version(
        car_model.feature_order,
        car_model.numeric_features,
        car_model.categorical_features,
        car_model.target_column,
    )
    if current_schema_version != car_model.schema_version:
        raise ValueError(
            f"Current code schema version {current_schema_version} does not match "
            f"model schema version {car_model.schema_version}. "
            f"Please use the same code version that was used for training."
        )
    print(f"✓ Current code schema version matches model")

    labels = {
        "mlflow_run_id": run_id,
        "data_version": params.get("data_version", "unknown"),
        "schema_version": car_model.schema_version,
        "model_type": params.get("model_type", "GradientBoostingRegressor"),
        "source": "mlflow_import",
    }

    saved_model = bentoml.sklearn.save(
        model_name,
        loaded_bundle,
        labels=labels,
        metadata=metadata,
    )

    print(f"\n{'=' * 60}")
    print(f"Import Complete")
    print(f"{'=' * 60}")
    print(f"BentoML tag: {saved_model.tag}")
    print(f"BentoML path: {saved_model.path}")
    print(f"\nLineage record:")
    print(f"  MLflow Run ID: {run_id}")
    print(f"  Data version: {params.get('data_version', 'N/A')}")
    print(f"  Schema version: {car_model.schema_version}")
    print(f"  R2 Score: {metrics.get('R2_Score', 'N/A')}")
    print(f"  BentoML tag: {saved_model.tag}")

    return {
        "bento_tag": str(saved_model.tag),
        "mlflow_run_id": run_id,
        "data_version": params.get("data_version"),
        "schema_version": car_model.schema_version,
        "metrics": dict(metrics),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Publish MLflow-trained model to BentoML (Single Source of Truth)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="MLflow run ID to import model from (only source of truth)",
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

    print("\n" + "!" * 60)
    print("IMPORTANT: This script ONLY imports models from MLflow.")
    print("Model training is the sole responsibility of mlflow/mlflow_app.py")
    print("This ensures training and deployment use the exact same code path.")
    print("!" * 60 + "\n")

    result = import_from_mlflow(
        run_id=args.run_id,
        model_name=args.model_name,
        tracking_uri=args.tracking_uri,
    )

    print(f"\n✓ Model published successfully")
    print(f"  Use 'bentoml models list' to see all saved models")
    print(f"  Use 'bentoml serve service:svc' to start the service")
    print(f"  Call /metadata API to verify lineage information")


if __name__ == "__main__":
    main()
