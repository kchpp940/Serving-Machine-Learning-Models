from __future__ import annotations

import os
from typing import Any, Dict
import numpy as np
import pandas as pd

from car_pricing.feature_schema import (
    FeatureSchema,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    bundle_model,
    is_model_bundle,
)

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


_DEFAULT_FEATURE_ORDER_LEGACY = list(FEATURE_ORDER)
_NUMERIC_FEATURE_SET = set(NUMERIC_FEATURES)
_CATEGORICAL_FEATURE_SET = set(CATEGORICAL_FEATURES)


def _is_legacy_bundle(obj) -> bool:
    return isinstance(obj, dict) and "model" in obj and "feature_order" in obj and "schema" not in obj


class CarPriceModel:
    """统一的推理适配器。

    支持三种格式（按优先级）：
    1. Schema Bundle（新格式）：{"model": ..., "schema": {...}}
    2. Legacy Bundle（旧格式）：{"model": ..., "feature_order": [...], "categorical_encoders": {...}}
    3. Legacy 裸模型：直接 sklearn 模型对象，回退到硬编码特征顺序
    """

    def __init__(self, raw):
        self._raw = raw
        self.mode = "unknown"
        self.model = None
        self.schema = None

        if is_model_bundle(raw):
            self.mode = "schema_bundle"
            self.model = raw["model"]
            self.schema = FeatureSchema.from_dict(raw["schema"])
        elif _is_legacy_bundle(raw):
            self.mode = "legacy_bundle"
            self.model = raw["model"]
            self.schema = self._schema_from_legacy_bundle(raw)
        else:
            self.mode = "legacy_raw"
            self.model = raw
            self.schema = self._build_legacy_schema()

        self._validate_dimensions()

    def _schema_from_legacy_bundle(self, raw: dict) -> FeatureSchema:
        schema = FeatureSchema(
            feature_order=list(raw["feature_order"]),
            target_column=raw.get("target_column", TARGET_COLUMN),
        )
        legacy_encoders = raw.get("categorical_encoders", {})
        if legacy_encoders:
            from sklearn.preprocessing import LabelEncoder
            encoders = {}
            for col, lb in legacy_encoders.items():
                encoders[col] = lb
            schema.categorical_encoders = encoders
        else:
            schema.categorical_encoders = self._build_legacy_encoders()
        return schema

    def _build_legacy_schema(self) -> FeatureSchema:
        schema = FeatureSchema()
        schema.categorical_encoders = self._build_legacy_encoders()
        return schema

    def _build_legacy_encoders(self):
        from sklearn.preprocessing import LabelEncoder

        encoders = {}
        legacy_classes = {
            "drivewheel": np.array(["4wd", "fwd", "rwd"]),
            "cylindernumber": np.array(
                ["eight", "five", "four", "six", "three", "twelve", "two"]
            ),
        }
        for col, classes in legacy_classes.items():
            lb = LabelEncoder()
            lb.fit(classes.astype(str))
            encoders[col] = lb
        return encoders

    def _validate_dimensions(self) -> None:
        expected = self.schema.n_features()
        actual = getattr(self.model, "n_features_in_", expected)
        if actual != expected:
            raise RuntimeError(
                f"模型 n_features_in_={actual} 与 schema 特征数 {expected} 不一致"
            )
        self.schema.validate_model_input(self.model)

    @classmethod
    def from_joblib(cls, path: str) -> "CarPriceModel":
        if joblib is None:
            raise RuntimeError("需要先安装 joblib")
        raw = joblib.load(path)
        return cls(raw)

    @classmethod
    def from_sklearn_object(cls, model) -> "CarPriceModel":
        if is_model_bundle(model) or _is_legacy_bundle(model):
            return cls(model)
        return cls(model)

    # ---------- 元数据 ----------

    @property
    def feature_order(self) -> list:
        return list(self.schema.feature_order)

    @property
    def numeric_features(self) -> list:
        return list(self.schema.numeric_features)

    @property
    def categorical_features(self) -> list:
        return list(self.schema.categorical_features)

    @property
    def target_column(self) -> str:
        return self.schema.target_column

    def categorical_classes(self, field_name: str):
        return self.schema.categorical_classes(field_name)

    def categorical_options(self, field_name: str) -> list:
        return self.schema.categorical_options(field_name)

    # ---------- 编码 ----------

    def encode_feature(self, field_name: str, value):
        return self.schema.encode_feature(field_name, value)

    def encode_dict(self, values: dict) -> dict:
        return self.schema.encode_dict(values)

    # ---------- 推理 ----------

    def _vector_from_encoded(self, encoded: dict) -> np.ndarray:
        return self.schema.vector_from_dict(encoded)

    def predict_encoded(self, encoded: dict) -> np.ndarray:
        data = self._vector_from_encoded(encoded)
        return self.model.predict(data)

    def predict_raw(self, values: dict) -> np.ndarray:
        encoded = self.encode_dict(values)
        return self.predict_encoded(encoded)

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        data = self.schema.vector_from_dataframe(df)
        return self.model.predict(data)

    # ---------- FastAPI 兼容层 ----------

    def predict_from_pydantic(self, data) -> np.ndarray:
        encoded = {f: self.encode_feature(f, getattr(data, f)) for f in self.feature_order}
        return self.predict_encoded(encoded)

    # ---------- 导出 bundle ----------

    def to_bundle(self) -> dict:
        return bundle_model(self.model, self.schema)

    # ---------- metadata ----------

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "feature_order": self.feature_order,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target_column": self.target_column,
            "categorical_options": {
                f: self.categorical_options(f) for f in self.categorical_features
            },
            "model_mode": self.mode,
            "n_features": self.schema.n_features(),
        }

    # ---------- self_check ----------

    def self_check(self) -> Dict[str, Any]:
        import time

        result: Dict[str, Any] = {
            "passed": True,
            "checks": [],
            "errors": [],
            "timestamp": time.time(),
        }

        check_schema = {"name": "schema_valid", "passed": False, "detail": ""}
        try:
            self.schema.validate()
            check_schema["passed"] = True
            check_schema["detail"] = "FeatureSchema validation passed"
        except Exception as e:
            check_schema["passed"] = False
            check_schema["detail"] = str(e)
            result["passed"] = False
            result["errors"].append(f"Schema validation failed: {e}")
        result["checks"].append(check_schema)

        check_dimensions = {"name": "model_dimensions_match", "passed": False, "detail": ""}
        try:
            expected = self.schema.n_features()
            actual = getattr(self.model, "n_features_in_", expected)
            if actual == expected:
                check_dimensions["passed"] = True
                check_dimensions["detail"] = f"n_features_in_={actual} matches schema features={expected}"
            else:
                check_dimensions["passed"] = False
                check_dimensions["detail"] = f"n_features_in_={actual} != schema features={expected}"
                result["passed"] = False
                result["errors"].append(
                    f"Model dimension mismatch: n_features_in_={actual}, schema features={expected}"
                )
        except Exception as e:
            check_dimensions["passed"] = False
            check_dimensions["detail"] = str(e)
            result["passed"] = False
            result["errors"].append(f"Dimension check error: {e}")
        result["checks"].append(check_dimensions)

        check_predict = {"name": "prediction_smoke_test", "passed": False, "detail": ""}
        try:
            sample = {}
            for f in self.schema.feature_order:
                if f in self.schema.numeric_features:
                    sample[f] = 0.0
                elif f in self.schema.categorical_features:
                    classes = self.schema.categorical_classes(f)
                    sample[f] = classes[0] if classes else ""
            pred = self.predict_raw(sample)
            if len(pred) == 1 and isinstance(float(pred[0]), float):
                check_predict["passed"] = True
                check_predict["detail"] = f"Smoke prediction succeeded, output={float(pred[0]):.4f}"
            else:
                check_predict["passed"] = False
                check_predict["detail"] = f"Unexpected prediction shape: {pred}"
                result["passed"] = False
                result["errors"].append(f"Smoke test returned unexpected output: {pred}")
        except Exception as e:
            check_predict["passed"] = False
            check_predict["detail"] = str(e)
            result["passed"] = False
            result["errors"].append(f"Prediction smoke test failed: {e}")
        result["checks"].append(check_predict)

        result["timestamp"] = time.time()
        return result
