import sys
import os
import json
import hashlib
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from car_pricing.feature_schema import (
    FeatureSchema,
    load_training_data,
    prepare_training_data,
    bundle_model,
    compute_schema_version,
    FEATURE_ORDER,
)


PRIMARY_METRIC = "r2_score"
HIGHER_IS_BETTER = True

CANDIDATE_MODELS = [
    {
        "name": "GradientBoosting",
        "class": GradientBoostingRegressor,
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
    },
    {
        "name": "RandomForest",
        "class": RandomForestRegressor,
        "params": {"n_estimators": 100, "max_depth": None, "random_state": 42},
    },
    {
        "name": "LinearRegression",
        "class": LinearRegression,
        "params": {},
    },
]


def compute_data_version(csv_path: str) -> str:
    with open(csv_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def _hash_artifact(run_id: str) -> str:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="model",
    )
    if not local_path or not os.path.exists(local_path):
        return None
    hasher = hashlib.sha256()
    if os.path.isdir(local_path):
        for root, dirs, files in os.walk(local_path):
            dirs.sort()
            for filename in sorted(files):
                filepath = os.path.join(root, filename)
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
    else:
        with open(local_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def train_single_candidate(model_name, model_class, model_params, X_train, X_test, y_train, y_test, schema, data_version, schema_version):
    with mlflow.start_run(run_name=model_name, nested=True) as run:
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        mlflow.log_param("model_name", model_name)
        for k, v in model_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("schema_version", schema_version)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("n_features", schema.n_features())
        mlflow.log_param("model_type", type(model).__name__)

        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)

        bundle = bundle_model(model, schema)
        mlflow.sklearn.log_model(bundle, "model")

        artifact_hash = _hash_artifact(run.info.run_id)
        if artifact_hash:
            mlflow.log_param("model_artifact_hash", artifact_hash)
            mlflow.set_tag("model_artifact_hash", artifact_hash)

        mlflow.set_tag("candidate_model", model_name)
        mlflow.set_tag("schema_version", schema_version)
        mlflow.set_tag("data_version", data_version)

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "model_type": type(model).__name__,
            "params": model_params,
            "metrics": {
                "r2_score": r2,
                "mse": mse,
                "mae": mae,
            },
            "schema_version": schema_version,
            "data_version": data_version,
            "model_artifact_hash": artifact_hash,
        }


def select_best_run(candidates, primary_metric, higher_is_better):
    if higher_is_better:
        best = max(candidates, key=lambda x: x["metrics"][primary_metric])
    else:
        best = min(candidates, key=lambda x: x["metrics"][primary_metric])
    return best


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "cars.csv")
    experiment_name = "car_price_candidates"
    tracking_uri = os.path.join(os.path.dirname(__file__), "mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    df = load_training_data(csv_path)
    X, y, schema = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    data_version = compute_data_version(csv_path)
    schema_version = compute_schema_version(schema)

    print(f"Data version: {data_version}")
    print(f"Schema version: {schema_version}")
    print(f"Number of features: {schema.n_features()}")

    with mlflow.start_run(run_name="candidate_comparison") as parent_run:
        mlflow.log_param("primary_metric", PRIMARY_METRIC)
        mlflow.log_param("higher_is_better", HIGHER_IS_BETTER)
        mlflow.log_param("n_candidates", len(CANDIDATE_MODELS))
        mlflow.log_param("schema_version", schema_version)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        candidates = []
        for cfg in CANDIDATE_MODELS:
            print(f"\n{'='*50}")
            print(f"Training candidate: {cfg['name']}")
            print(f"Params: {cfg['params']}")

            result = train_single_candidate(
                cfg["name"],
                cfg["class"],
                cfg["params"],
                X_train, X_test, y_train, y_test,
                schema,
                data_version,
                schema_version,
            )
            candidates.append(result)

            print(f"R2 Score: {result['metrics']['r2_score']:.4f}")
            print(f"MSE: {result['metrics']['mse']:.4f}")
            print(f"MAE: {result['metrics']['mae']:.4f}")

        best = select_best_run(candidates, PRIMARY_METRIC, HIGHER_IS_BETTER)

        print(f"\n{'='*50}")
        print(f"Best model: {best['model_name']}")
        print(f"Best {PRIMARY_METRIC}: {best['metrics'][PRIMARY_METRIC]:.4f}")
        print(f"Run ID: {best['run_id']}")

        summary = {
            "primary_metric": PRIMARY_METRIC,
            "higher_is_better": HIGHER_IS_BETTER,
            "best_model": best["model_name"],
            "best_run_id": best["run_id"],
            "best_metric_value": best["metrics"][PRIMARY_METRIC],
            "schema_version": schema_version,
            "data_version": data_version,
            "best_model_artifact_hash": best.get("model_artifact_hash"),
            "best_model_type": best.get("model_type"),
            "candidates": [
                {
                    "model_name": c["model_name"],
                    "model_type": c.get("model_type"),
                    "run_id": c["run_id"],
                    "metrics": c["metrics"],
                    "params": c["params"],
                    "model_artifact_hash": c.get("model_artifact_hash"),
                }
                for c in candidates
            ],
        }

        summary_json = json.dumps(summary, indent=2)
        summary_path = os.path.join(os.path.dirname(__file__), "candidate_summary.json")
        with open(summary_path, "w") as f:
            f.write(summary_json)

        mlflow.log_artifact(summary_path)
        mlflow.log_dict(summary, "candidate_summary.json")
        mlflow.log_param("candidate_summary", summary_json)
        mlflow.log_param("best_run_id", best["run_id"])
        mlflow.log_param("best_model", best["model_name"])
        mlflow.log_param("best_metric_value", best["metrics"][PRIMARY_METRIC])
        mlflow.set_tag("best_model", best["model_name"])
        mlflow.set_tag("best_run_id", best["run_id"])
        mlflow.set_tag("parent_run_type", "candidate_comparison")

        for c in candidates:
            mlflow.log_metric(f"{c['model_name']}_r2_score", c["metrics"]["r2_score"])
            mlflow.log_metric(f"{c['model_name']}_mse", c["metrics"]["mse"])
            mlflow.log_metric(f"{c['model_name']}_mae", c["metrics"]["mae"])

        print(f"\n{'='*50}")
        print(f"Parent run ID: {parent_run.info.run_id}")
        print(f"Tracking URI: {tracking_uri}")
        print(f"Best model: {best['model_name']} (run: {best['run_id']})")
        print(f"Use with publish script:")
        print(f"  python publish_best_run.py --parent-run-id {parent_run.info.run_id}")
        print(f"  or: python publish_best_run.py --best-run-id {best['run_id']}")

        return summary, parent_run.info.run_id, tracking_uri


if __name__ == "__main__":
    main()
