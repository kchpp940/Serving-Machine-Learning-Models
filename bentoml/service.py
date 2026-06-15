import sys
import os
import importlib.util

_bento_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = os.path.abspath(os.path.join(_bento_dir, ".."))

_bentoml_safe_path = os.path.join(_project_root, "car_pricing", "bentoml_safe.py")
_bentoml_spec = importlib.util.spec_from_file_location(
    "car_pricing_bentoml_safe",
    _bentoml_safe_path,
)
_bentoml_safe_module = importlib.util.module_from_spec(_bentoml_spec)
sys.modules["car_pricing_bentoml_safe"] = _bentoml_safe_module
_bentoml_spec.loader.exec_module(_bentoml_safe_module)

bentoml = _bentoml_safe_module.bentoml
picklable_model = _bentoml_safe_module.picklable_model
NumpyNdarray = _bentoml_safe_module.NumpyNdarray
PandasDataFrame = _bentoml_safe_module.PandasDataFrame
JSON = _bentoml_safe_module.JSON
Service = _bentoml_safe_module.Service

import numpy as np
import pandas as pd

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER


BENTO_MODEL_NAME = "car_price_model"

predictor = picklable_model.get(BENTO_MODEL_NAME + ":latest").to_runner()

svc = Service(BENTO_MODEL_NAME, runners=[predictor])


def _desanitize_value(val):
    if val == "null":
        return None
    if isinstance(val, dict):
        return {k: _desanitize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_desanitize_value(v) for v in val]
    return val


def _get_model_bundle():
    bento_model = picklable_model.get(BENTO_MODEL_NAME + ":latest")
    raw_bundle = picklable_model.load_model(BENTO_MODEL_NAME + ":latest")
    return raw_bundle, bento_model


_model = None
_model_metadata = None


def get_model():
    global _model
    if _model is None:
        raw_bundle, _ = _get_model_bundle()
        _model = CarPriceModel.from_sklearn_object(raw_bundle)
        _model.schema.validate()
    return _model


def get_model_metadata():
    global _model_metadata
    if _model_metadata is None:
        _, bento_model = _get_model_bundle()
        raw_meta = bento_model.info.metadata or {}
        _model_metadata = _desanitize_value(raw_meta)
    return _model_metadata


@svc.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.ndarray:
    model = get_model()

    missing_cols = set(FEATURE_ORDER) - set(df.columns)
    if missing_cols:
        raise ValueError(f"输入数据缺少列: {missing_cols}")

    extra_cols = set(df.columns) - set(FEATURE_ORDER)
    if extra_cols:
        df = df[FEATURE_ORDER]

    result = model.predict_dataframe(df)
    return np.array(result)


@svc.api(input=JSON(), output=JSON())
def metadata(_) -> dict:
    model = get_model()
    meta = get_model_metadata()

    candidate_info = meta.get("candidate_info", {})
    consistency_check = meta.get("consistency_check", {})

    response = {
        "model_name": BENTO_MODEL_NAME,
        "schema": {
            "feature_order": model.feature_order,
            "numeric_features": model.numeric_features,
            "categorical_features": model.categorical_features,
            "target_column": model.target_column,
            "n_features": model.schema.n_features(),
        },
        "categorical_options": {
            col: model.categorical_options(col)
            for col in model.categorical_features
        },
    }

    if candidate_info:
        candidates = candidate_info.get("candidates", [])
        best_model = candidate_info.get("best_model")
        primary_metric = candidate_info.get("primary_metric", "r2_score")
        higher_is_better = candidate_info.get("higher_is_better", True)

        def _metric_value(c):
            v = c.get("metrics", {}).get(primary_metric, 0)
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0

        candidates_sorted = sorted(
            candidates,
            key=_metric_value,
            reverse=higher_is_better,
        )

        candidate_summary = []
        for rank, c in enumerate(candidates_sorted, 1):
            is_best = c.get("model_name") == best_model
            candidate_summary.append({
                "rank": rank,
                "model_name": c.get("model_name"),
                "is_best": is_best,
                "run_id": c.get("run_id"),
                "metrics": c.get("metrics", {}),
                "params": c.get("params", {}),
            })

        response["deployment"] = {
            "best_model": best_model,
            "best_run_id": candidate_info.get("best_run_id"),
            "parent_run_id": candidate_info.get("parent_run_id"),
            "primary_metric": primary_metric,
            "higher_is_better": higher_is_better,
            "best_metric_value": candidate_info.get("best_metric_value"),
            "schema_version": candidate_info.get("schema_version"),
            "data_version": candidate_info.get("data_version"),
            "candidate_summary": candidate_summary,
        }

        if consistency_check and consistency_check.get("performed"):
            response["deployment"]["consistency_check"] = {
                "performed": True,
                "all_checks_passed": consistency_check.get("all_checks_passed"),
                "model_artifact_hash": consistency_check.get("model_artifact_hash"),
                "verified_at": consistency_check.get("verified_at"),
                "checks": consistency_check.get("checks", {}),
            }
        else:
            response["deployment"]["consistency_check"] = {
                "performed": False,
                "reason": consistency_check.get("reason", "Consistency check not performed"),
            }
    else:
        response["deployment"] = {
            "note": "此模型未通过候选选择流程发布，缺少候选对比信息",
        }

    return response
