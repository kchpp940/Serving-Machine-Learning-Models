import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
import mlflow
import bentoml

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import (
    calculate_schema_version,
    diff_schema_versions,
    prepare_training_data,
    load_training_data,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    SCHEMA_FORMAT_VERSION,
)


class SchemaVersionMismatchError(Exception):
    pass


def _build_current_code_schema_dict():
    csv_path = os.path.join(os.path.dirname(__file__), "Data", "cars.csv")
    df = load_training_data(csv_path)
    _, _, schema = prepare_training_data(df)
    return schema.schema_dict_for_version


def _validate_schema_version_triple(
    bundle_schema_dict: dict,
    mlflow_schema_version: str,
    current_code_schema_dict: dict,
):
    errors = []

    bundle_sv = calculate_schema_version(bundle_schema_dict)
    code_sv = calculate_schema_version(current_code_schema_dict)

    sources = {
        "bundle": bundle_sv,
        "mlflow_params": mlflow_schema_version,
        "current_code": code_sv,
    }

    if len(set(filter(None, sources.values()))) <= 1:
        return

    pair_checks = [
        ("bundle", "mlflow_params"),
        ("bundle", "current_code"),
        ("mlflow_params", "current_code"),
    ]

    all_diffs = []
    for src_a, src_b in pair_checks:
        sv_a = sources[src_a]
        sv_b = sources[src_b]
        if sv_a and sv_b and sv_a != sv_b:
            if src_a == "bundle" and src_b == "current_code":
                diffs = diff_schema_versions(
                    bundle_schema_dict, src_a,
                    current_code_schema_dict, src_b,
                )
            elif src_a == "mlflow_params" and src_b == "current_code":
                diffs = ["(MLflow params only stores version hash, no dict to diff)"]
            elif src_a == "bundle" and src_b == "mlflow_params":
                diffs = ["(MLflow params only stores version hash, no dict to diff)"]
            else:
                diffs = []
            all_diffs.append(f"  {src_a}={sv_a} vs {src_b}={sv_b}")
            if diffs:
                for d in diffs:
                    all_diffs.append(f"    - {d}")

    detail = "\n".join(all_diffs)
    raise SchemaVersionMismatchError(
        f"Schema version mismatch detected across sources!\n"
        f"  bundle:          {sources['bundle']}\n"
        f"  mlflow_params:   {sources['mlflow_params']}\n"
        f"  current_code:    {sources['current_code']}\n"
        f"Differences:\n{detail}\n"
        f"ABORTING: All three sources must agree on schema_version."
    )


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
    print(f"Schema version (bundle): {car_model.schema_version}")
    print(f"Feature order: {car_model.feature_order}")
    print(f"Number of features: {car_model.schema.n_features()}")

    mlflow_schema_version = params.get("schema_version")
    bundle_schema_dict = car_model.schema_dict_for_version

    print(f"\n--- Triple schema version validation ---")
    print(f"  bundle schema_version:      {car_model.schema_version}")
    print(f"  mlflow_params schema_version: {mlflow_schema_version}")

    try:
        current_code_schema_dict = _build_current_code_schema_dict()
        current_code_sv = calculate_schema_version(current_code_schema_dict)
        print(f"  current_code schema_version: {current_code_sv}")
    except Exception as e:
        print(f"  WARNING: Could not compute current code schema_version: {e}")
        current_code_schema_dict = None
        current_code_sv = None

    if mlflow_schema_version and car_model.schema_version != mlflow_schema_version:
        diffs = diff_schema_versions(
            bundle_schema_dict, "bundle",
            current_code_schema_dict or bundle_schema_dict, "reference",
        )
        diff_detail = "\n  ".join(diffs) if diffs else "no detailed diff available"
        raise SchemaVersionMismatchError(
            f"Bundle schema_version ({car_model.schema_version}) != "
            f"MLflow params schema_version ({mlflow_schema_version})\n"
            f"Diff details:\n  {diff_detail}\n"
            f"ABORTING: Bundle and MLflow records disagree."
        )
    print(f"  ✓ bundle == mlflow_params")

    if current_code_sv and car_model.schema_version != current_code_sv:
        diffs = diff_schema_versions(
            bundle_schema_dict, "bundle",
            current_code_schema_dict, "current_code",
        )
        diff_detail = "\n  ".join(diffs) if diffs else "no detailed diff available"
        raise SchemaVersionMismatchError(
            f"Bundle schema_version ({car_model.schema_version}) != "
            f"Current code schema_version ({current_code_sv})\n"
            f"Diff details:\n  {diff_detail}\n"
            f"ABORTING: Current code has drifted from the training code. "
            f"Use the same code version that was used for training."
        )
    if current_code_sv:
        print(f"  ✓ bundle == current_code")

    training_metadata = None
    try:
        metadata_artifact = client.download_artifacts(
            run_id, "metadata/training_metadata.json"
        )
        with open(metadata_artifact, "r", encoding="utf-8") as f:
            training_metadata = json.load(f)
        print("✓ Loaded training metadata from MLflow")

        tm_sv = training_metadata.get("schema_version")
        if tm_sv and tm_sv != car_model.schema_version:
            raise SchemaVersionMismatchError(
                f"training_metadata schema_version ({tm_sv}) != "
                f"bundle schema_version ({car_model.schema_version})\n"
                f"ABORTING: Training metadata and bundle disagree."
            )
        print(f"  ✓ training_metadata schema_version matches bundle")

        tm_feature_schema = training_metadata.get("feature_schema", {})
        tm_sv_from_fs = tm_feature_schema.get("schema_version")
        if tm_sv_from_fs and tm_sv_from_fs != car_model.schema_version:
            raise SchemaVersionMismatchError(
                f"training_metadata.feature_schema.schema_version ({tm_sv_from_fs}) != "
                f"bundle schema_version ({car_model.schema_version})\n"
                f"ABORTING: Nested feature_schema version disagrees."
            )
        print(f"  ✓ training_metadata.feature_schema schema_version matches bundle")

    except SchemaVersionMismatchError:
        raise
    except Exception as e:
        print(f"Note: Could not load full training metadata: {e}")

    print(f"\n✓ All schema version checks PASSED: {car_model.schema_version}")

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

    try:
        result = import_from_mlflow(
            run_id=args.run_id,
            model_name=args.model_name,
            tracking_uri=args.tracking_uri,
        )

        print(f"\n✓ Model published successfully")
        print(f"  Use 'bentoml models list' to see all saved models")
        print(f"  Use 'bentoml serve service:svc' to start the service")
        print(f"  Call /metadata API to verify lineage information")
    except SchemaVersionMismatchError as e:
        print(f"\n{'✗' * 3} PUBLISH ABORTED {'✗' * 3}")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
