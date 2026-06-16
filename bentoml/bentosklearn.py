import os
import tempfile
import joblib

import bentoml
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from car_pricing.feature_schema import (
    load_training_data,
    prepare_training_data,
    bundle_model,
)
from car_pricing.model_lineage import (
    record_training_lineage,
    build_bentoml_metadata,
)


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "Data", "cars.csv")
    df = load_training_data(csv_path)

    X, y, schema = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model_name = "gbr"
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

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_model_path = f.name
    joblib.dump(bundle, temp_model_path)

    lineage = record_training_lineage(
        model_name=model_name,
        model_type=model_type,
        model=model,
        schema=schema,
        csv_path=csv_path,
        model_path=temp_model_path,
        metrics=metrics,
        save_metadata=False,
    )

    bentoml_metadata = build_bentoml_metadata(lineage)

    labels = {
        "model_type": lineage.model_type,
        "schema_version": lineage.schema_version,
        "data_version": lineage.data_version,
        "model_artifact_hash": lineage.model_artifact_hash,
    }

    bentoml.sklearn.save(
        model_name,
        bundle,
        labels=labels,
        metadata=bentoml_metadata,
    )

    try:
        os.unlink(temp_model_path)
    except OSError:
        pass

    print(f"Model saved to BentoML")
    print(f"Model name: {lineage.model_name}")
    print(f"Model type: {lineage.model_type}")
    print(f"Feature order: {schema.feature_order}")
    print(f"Number of features: {schema.n_features()}")
    print(f"Model n_features_in_: {model.n_features_in_}")
    print(f"Schema version: {lineage.schema_version}")
    print(f"Data version: {lineage.data_version}")
    print(f"Model artifact hash: {lineage.model_artifact_hash}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
