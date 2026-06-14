import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from car_pricing.feature_schema import (
    load_training_data,
    prepare_training_data,
    bundle_model,
    calculate_data_version,
)


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "cars.csv")
    data_version = calculate_data_version(csv_path)
    data_abs_path = os.path.abspath(csv_path)

    with mlflow.start_run():
        mlflow.log_param("data_path", data_abs_path)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("data_file", os.path.basename(csv_path))

        df = load_training_data(csv_path)
        mlflow.log_param("data_shape", str(df.shape))
        mlflow.log_param("num_samples", len(df))

        mlflow.log_artifact(csv_path, "data")

        print(f'data shape is: {df.shape}')
        print(f'data version: {data_version}')

        X, y, schema = prepare_training_data(df)

        schema_dict = schema.to_dict()
        schema_json = json.dumps(schema_dict, indent=2, ensure_ascii=False)
        schema_path = os.path.join(os.path.dirname(__file__), "feature_schema.json")
        with open(schema_path, "w", encoding="utf-8") as f:
            f.write(schema_json)
        mlflow.log_artifact(schema_path, "metadata")
        mlflow.log_dict(schema_dict, "metadata/feature_schema.json")

        mlflow.log_param("feature_order", str(schema.feature_order))
        mlflow.log_param("numeric_features", str(schema.numeric_features))
        mlflow.log_param("categorical_features", str(schema.categorical_features))
        mlflow.log_param("target_column", schema.target_column)

        for col, encoder in schema.categorical_encoders.items():
            mlflow.log_param(f"encoder_{col}_classes", str(list(encoder.classes_)))

        print(f'Feature order: {schema.feature_order}')
        print(f'Number of features: {schema.n_features()}')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        model_params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        model = GradientBoostingRegressor(**model_params)

        for param_name, param_value in model_params.items():
            mlflow.log_param(f"model_{param_name}", param_value)

        model.fit(X_train, y_train)

        schema.validate_model_input(model)
        mlflow.log_param("model_n_features_in", model.n_features_in_)

        print(f'GBR model is training')

        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        mlflow.log_metric('R2_Score', r2)
        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('RMSE', mse ** 0.5)

        print(f'R2 Score: {r2*100:.4f}%')
        print(f'MSE: {mse:.4f}')
        print(f'MAE: {mae:.4f}')

        bundle = bundle_model(model, schema)

        mlflow.sklearn.log_model(bundle, 'model')

        bundle_path = os.path.join(os.path.dirname(__file__), "model_bundle.pkl")
        import joblib
        joblib.dump(bundle, bundle_path)
        mlflow.log_artifact(bundle_path, "model_bundle")

        training_metadata = {
            "data_path": data_abs_path,
            "data_version": data_version,
            "data_shape": [df.shape[0], df.shape[1]],
            "feature_schema": schema_dict,
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
                "random_state": 1,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            },
        }
        mlflow.log_dict(training_metadata, "metadata/training_metadata.json")

        metadata_path = os.path.join(os.path.dirname(__file__), "training_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(training_metadata, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(metadata_path, "metadata")

        if os.path.exists(schema_path):
            os.remove(schema_path)
        if os.path.exists(bundle_path):
            os.remove(bundle_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model and metadata logged to MLflow successfully")


if __name__ == "__main__":
    main()