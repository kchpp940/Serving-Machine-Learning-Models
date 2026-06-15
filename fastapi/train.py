import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

from car_pricing.feature_schema import (
    load_training_data,
    prepare_training_data,
    bundle_model,
)
from car_pricing.versioning import (
    compute_schema_version,
    compute_data_version,
    compute_artifact_hash,
)


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "Data", "cars.csv")
    df = load_training_data(csv_path)

    X, y, schema = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    schema.validate_model_input(model)

    bundle = bundle_model(model, schema)

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sklearn_gbr.pkl")
    joblib.dump(bundle, model_path)

    data_version = compute_data_version(csv_path)
    schema_version = compute_schema_version(schema)
    artifact_hash = compute_artifact_hash(bundle)

    print(f"Model saved to {model_path}")
    print(f"Feature order: {schema.feature_order}")
    print(f"Number of features: {schema.n_features()}")
    print(f"Model n_features_in_: {model.n_features_in_}")
    print(f"data_version: {data_version}")
    print(f"schema_version: {schema_version}")
    print(f"model_artifact_hash: {artifact_hash}")


if __name__ == "__main__":
    main()
