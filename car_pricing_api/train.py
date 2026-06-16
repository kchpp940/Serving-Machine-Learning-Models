import os

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
from car_pricing.model_lineage import (
    record_training_lineage,
)


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "Data", "cars.csv")
    df = load_training_data(csv_path)

    X, y, schema = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model_name = "sklearn_gbr"
    model_type = "GradientBoostingRegressor"
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    schema.validate_model_input(model)

    bundle = bundle_model(model, schema)

    y_pred = model.predict(X_test)
    metrics = {
        "r2_score": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sklearn_gbr.pkl")
    joblib.dump(bundle, model_path)

    lineage = record_training_lineage(
        model_name=model_name,
        model_type=model_type,
        model=model,
        schema=schema,
        csv_path=csv_path,
        model_path=model_path,
        metrics=metrics,
        save_metadata=True,
        metadata_dir=model_dir,
    )

    print(f"Model saved to {model_path}")
    print(f"Model name: {lineage.model_name}")
    print(f"Model type: {lineage.model_type}")
    print(f"Feature order: {schema.feature_order}")
    print(f"Number of features: {schema.n_features()}")
    print(f"Model n_features_in_: {model.n_features_in_}")
    print(f"Schema version: {lineage.schema_version}")
    print(f"Data version: {lineage.data_version}")
    print(f"Model artifact hash: {lineage.model_artifact_hash}")
    print(f"Metrics: {metrics}")
    print(f"Metadata saved to {os.path.join(model_dir, 'model_metadata.json')}")
    print(f"Status saved to {os.path.join(model_dir, 'model_status.json')}")


if __name__ == "__main__":
    main()
