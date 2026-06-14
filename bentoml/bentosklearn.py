import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bentoml
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from car_pricing.feature_schema import (
    load_training_data,
    prepare_training_data,
    bundle_model,
    calculate_data_version,
)


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "Data", "cars.csv")
    data_version = calculate_data_version(csv_path)
    data_abs_path = os.path.abspath(csv_path)

    print(f"Data path: {data_abs_path}")
    print(f"Data version: {data_version}")

    df = load_training_data(csv_path)
    print(f"Data shape: {df.shape}")

    X, y, schema = prepare_training_data(df)

    print(f"Feature order: {schema.feature_order}")
    print(f"Number of features: {schema.n_features()}")

    model_params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "random_state": 42,
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = GradientBoostingRegressor(**model_params)
    model.fit(X_train, y_train)

    schema.validate_model_input(model)
    print(f"Model n_features_in_: {model.n_features_in_}")

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print(f"R2 Score: {r2*100:.4f}%")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    bundle = bundle_model(model, schema)

    training_metadata = {
        "data_path": data_abs_path,
        "data_version": data_version,
        "data_shape": [df.shape[0], df.shape[1]],
        "feature_schema": schema.to_dict(),
        "model_type": "GradientBoostingRegressor",
        "model_params": model_params,
        "metrics": {
            "R2_Score": r2,
            "MSE": mse,
            "MAE": mae,
            "RMSE": mse ** 0.5,
        },
        "train_test_split": {
            "test_size": 0.2,
            "random_state": 0,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
    }

    bentoml.sklearn.save(
        "gbr",
        bundle,
        labels={
            "data_version": data_version,
            "model_type": "GradientBoostingRegressor",
            "source": "bentoml_training",
        },
        metadata={
            "training_metadata": training_metadata,
            "feature_schema": schema.to_dict(),
            "data_version": data_version,
            "data_path": data_abs_path,
            "mlflow_params": {
                "data_version": data_version,
                "data_path": data_abs_path,
                "feature_order": str(schema.feature_order),
            },
            "mlflow_metrics": {
                "R2_Score": r2,
                "MSE": mse,
                "MAE": mae,
                "RMSE": mse ** 0.5,
            },
        },
    )

    print(f"\nModel saved to BentoML with full metadata")
    print(f"Data version: {data_version}")
    print(f"Labels and metadata recorded for lineage tracking")


if __name__ == "__main__":
    main()
