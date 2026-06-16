import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

from car_pricing.feature_schema import (
    load_training_data,
    prepare_training_data,
    bundle_model,
)
from car_pricing.config import load_config
from car_pricing.versioning import compute_data_version, compute_file_hash
from car_pricing.model_lineage import (
    ModelLineage,
    build_fastapi_status,
)


def main():
    config = load_config()

    csv_path = config.data_csv_path
    df = load_training_data(csv_path)

    X, y, schema = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model_name = "sklearn_gbr"
    model_type = "GradientBoostingRegressor"
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    schema.validate_model_input(model)

    bundle = bundle_model(model, schema)

    data_version = compute_data_version(csv_path)
    schema_version = schema.schema_version()

    y_pred = model.predict(X_test)
    metrics = {
        "r2_score": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    model_dir = config.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model_path = config.model_path
    joblib.dump(bundle, model_path)

    model_artifact_hash = compute_file_hash(model_path)

    lineage = ModelLineage(
        run_id="",
        experiment_id="",
        model_name=model_name,
        model_type=model_type,
        schema_version=schema_version,
        data_version=data_version,
        model_artifact_hash=model_artifact_hash,
        metrics=metrics,
        params={
            "model_name": model_name,
            "model_type": model_type,
            "n_features": schema.n_features(),
        },
        schema=schema,
    )

    metadata_path = config.model_metadata_path
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(lineage.to_dict(), f, indent=2, ensure_ascii=False)

    status_path = config.model_status_path
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(build_fastapi_status(lineage), f, indent=2, ensure_ascii=False)

    print(f"Model saved to {model_path}")
    print(f"Model name: {model_name}")
    print(f"Model type: {model_type}")
    print(f"Feature order: {schema.feature_order}")
    print(f"Number of features: {schema.n_features()}")
    print(f"Model n_features_in_: {model.n_features_in_}")
    print(f"Schema version: {schema_version}")
    print(f"Data version: {data_version}")
    print(f"Model artifact hash: {model_artifact_hash}")
    print(f"Metrics: {metrics}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Status saved to {status_path}")


if __name__ == "__main__":
    main()
