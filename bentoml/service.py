import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bentoml
import bentoml.sklearn
from bentoml.io import NumpyNdarray, PandasDataFrame, JSON

import numpy as np
import pandas as pd

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import (
    calculate_schema_version,
    diff_schema_versions,
    prepare_training_data,
    load_training_data,
)


_model_tag = "gbr:latest"
predictor = bentoml.sklearn.load_runner(_model_tag)

service = bentoml.Service("gbr", runners=[predictor])


_bento_model_info = None


def _load_model_and_metadata():
    global _bento_model_info
    bento_model = bentoml.sklearn.get(_model_tag)
    _bento_model_info = {
        "tag": str(bento_model.tag),
        "path": bento_model.path,
        "version": bento_model.tag.version,
    }
    raw_bundle = bento_model.to_sklearn()
    model = CarPriceModel.from_sklearn_object(raw_bundle)
    metadata = bento_model.info.metadata
    labels = bento_model.info.labels
    return model, metadata, labels


_model = None
_model_metadata = None
_model_labels = None


def get_model():
    global _model, _model_metadata, _model_labels
    if _model is None:
        _model, _model_metadata, _model_labels = _load_model_and_metadata()
        _model.schema.validate()
    return _model


def get_model_metadata():
    if _model is None:
        get_model()
    return _model_metadata, _model_labels


def _get_lineage_info():
    model = get_model()
    metadata, labels = get_model_metadata()

    mlflow_params = metadata.get("mlflow_params", {}) if metadata else {}
    mlflow_metrics = metadata.get("mlflow_metrics", {}) if metadata else {}
    training_metadata = metadata.get("training_metadata", {}) if metadata else {}

    run_id_from_metadata = metadata.get("mlflow_run_id") if metadata else None
    run_id_from_labels = labels.get("mlflow_run_id") if labels else None
    run_id_from_training = training_metadata.get("mlflow_run_id")

    mlflow_run_id = run_id_from_metadata or run_id_from_labels or run_id_from_training

    data_version_from_labels = labels.get("data_version") if labels else None
    data_version_from_params = mlflow_params.get("data_version")
    data_version_from_training = training_metadata.get("data_version")
    data_version = data_version_from_labels or data_version_from_params or data_version_from_training

    schema_version_from_model = model.schema_version
    schema_version_from_labels = labels.get("schema_version") if labels else None
    schema_version_from_params = mlflow_params.get("schema_version")
    schema_version_from_training = training_metadata.get("schema_version")
    schema_version_from_feature_schema = (
        metadata.get("feature_schema", {}).get("schema_version") if metadata else None
    )
    schema_version = schema_version_from_labels or schema_version_from_params or schema_version_from_training

    metrics_from_mlflow = dict(mlflow_metrics)
    metrics_from_training = training_metadata.get("metrics", {})
    metrics = metrics_from_training if metrics_from_training else metrics_from_mlflow

    bundle_schema_dict = model.schema_dict_for_version

    consistency_checks = {
        "mlflow_run_id": {
            "from_metadata": run_id_from_metadata,
            "from_labels": run_id_from_labels,
            "consistent": (run_id_from_metadata == run_id_from_labels) if (run_id_from_metadata and run_id_from_labels) else True,
        },
        "data_version": {
            "from_labels": data_version_from_labels,
            "from_mlflow_params": data_version_from_params,
            "from_training_metadata": data_version_from_training,
            "consistent": len(set(filter(None, [
                data_version_from_labels,
                data_version_from_params,
                data_version_from_training
            ]))) <= 1,
        },
        "schema_version": {
            "from_model_calculated": schema_version_from_model,
            "from_labels": schema_version_from_labels,
            "from_mlflow_params": schema_version_from_params,
            "from_training_metadata": schema_version_from_training,
            "from_feature_schema_nested": schema_version_from_feature_schema,
            "consistent": len(set(filter(None, [
                schema_version_from_model,
                schema_version_from_labels,
                schema_version_from_params,
                schema_version_from_training,
                schema_version_from_feature_schema,
            ]))) <= 1,
        },
    }

    schema_diffs = []
    tm_feature_schema = training_metadata.get("feature_schema", {})
    if tm_feature_schema and bundle_schema_dict != tm_feature_schema:
        schema_diffs = diff_schema_versions(
            tm_feature_schema, "training_metadata.feature_schema",
            bundle_schema_dict, "bundle",
        )

    try:
        csv_path = os.path.join(os.path.dirname(__file__), "Data", "cars.csv")
        df = load_training_data(csv_path)
        _, _, current_schema = prepare_training_data(df)
        current_code_schema_dict = current_schema.schema_dict_for_version
        current_code_sv = current_schema.schema_version
        consistency_checks["schema_version"]["from_current_code"] = current_code_sv
        if current_code_sv != schema_version_from_model:
            consistency_checks["schema_version"]["consistent"] = False
            code_diffs = diff_schema_versions(
                current_code_schema_dict, "current_code",
                bundle_schema_dict, "bundle",
            )
            schema_diffs.extend(code_diffs)
    except Exception:
        pass

    all_consistent = all(
        check["consistent"] for check in consistency_checks.values()
    )

    return {
        "bento_tag": _bento_model_info["tag"] if _bento_model_info else None,
        "bento_version": _bento_model_info["version"] if _bento_model_info else None,
        "mlflow_run_id": mlflow_run_id,
        "data_version": data_version,
        "schema_version": schema_version,
        "metrics": metrics,
        "consistency_checks": consistency_checks,
        "schema_diffs": schema_diffs,
        "all_consistent": all_consistent,
        "model_mode": model.mode,
        "source": labels.get("source") if labels else None,
    }


@service.api(input=JSON(), output=JSON())
def metadata(_) -> dict:
    model = get_model()
    metadata, labels = get_model_metadata()

    lineage = _get_lineage_info()

    result = {
        "service": "Car Price Prediction API",
        "lineage": {
            "bento_tag": lineage["bento_tag"],
            "bento_version": lineage["bento_version"],
            "mlflow_run_id": lineage["mlflow_run_id"],
            "data_version": lineage["data_version"],
            "schema_version": lineage["schema_version"],
            "metrics": lineage["metrics"],
        },
        "consistency": {
            "all_checks_passed": lineage["all_consistent"],
            "details": lineage["consistency_checks"],
            "schema_diffs": lineage["schema_diffs"],
        },
        "model_info": {
            "mode": model.mode,
            "feature_order": model.feature_order,
            "numeric_features": model.numeric_features,
            "categorical_features": model.categorical_features,
            "target_column": model.target_column,
            "schema_version": model.schema_version,
            "format_version": model.schema.format_version,
            "n_features": model.schema.n_features(),
        },
        "categorical_encoders": {
            col: {
                "classes": list(model.schema.categorical_encoders[col].classes_),
                "options": model.categorical_options(col),
            }
            for col in model.categorical_features
            if col in model.schema.categorical_encoders
        },
        "labels": labels or {},
        "raw_metadata": metadata or {},
    }

    return result


@service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.ndarray:
    model = get_model()
    feature_order = model.feature_order

    missing_cols = set(feature_order) - set(df.columns)
    if missing_cols:
        raise ValueError(f"输入数据缺少列: {missing_cols}")

    extra_cols = set(df.columns) - set(feature_order)
    if extra_cols:
        df = df[feature_order]

    result = model.predict_dataframe(df)
    return np.array(result)
