from __future__ import annotations

from typing import List, Sequence
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
    SchemaMismatchError,
    validate_service_schema,
)

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


def _is_legacy_bundle(obj) -> bool:
    return (
        isinstance(obj, dict)
        and "model" in obj
        and "feature_order" in obj
        and "schema" not in obj
    )


class CarPriceModel:
    """统一的推理适配器。

    支持三种格式（按优先级）：
    1. Schema Bundle（新格式）：{"model": ..., "schema": {...}}
    2. Legacy Bundle（旧格式）：{"model": ..., "feature_order": [...], ...}
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
            raise SchemaMismatchError(
                "model_n_features",
                f"模型 n_features_in_={actual} 与 schema 特征数 {expected} 不一致",
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
        return cls(model)

    def validate_service(self, interface_fields: Sequence[str]) -> None:
        errors = validate_service_schema(
            self.schema, self.model, interface_fields
        )
        if errors:
            msg = "服务 schema 校验失败:\n" + "\n".join(f"  - {e}" for e in errors)
            raise SchemaMismatchError("service_validation", msg)

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

    def model_info(self) -> dict:
        info = {
            "mode": self.mode,
            "n_features_in_": getattr(self.model, "n_features_in_", None),
            "schema_n_features": self.schema.n_features(),
            "feature_order": self.feature_order,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target_column": self.target_column,
        }
        info["schema"] = self.schema_export()
        info["form_fields"] = self.form_fields_metadata()
        info["input_spec"] = self.input_spec()
        return info

    def schema_export(self) -> dict:
        return self.schema.schema_export()

    def form_fields_metadata(self, extra_fields: list = None) -> list:
        return self.schema.form_fields_metadata(extra_fields=extra_fields)

    def input_spec(self) -> dict:
        return self.schema.input_spec()

    def default_example(self) -> dict:
        return self.schema.default_example()

    def field_label(self, field_name: str) -> str:
        return self.schema.field_label(field_name)

    def field_description(self, field_name: str) -> str:
        return self.schema.field_description(field_name)

    def field_allowed_values(self, field_name: str) -> list:
        return self.schema.field_allowed_values(field_name)

    def field_error_message(self, field_name: str, error_type: str) -> str:
        return self.schema.field_error_message(field_name, error_type)

    def categorical_classes(self, field_name: str) -> list:
        return self.schema.categorical_classes(field_name)

    def categorical_options(self, field_name: str) -> list:
        return self.schema.categorical_options(field_name)

    # ---------- 编码 ----------

    def encode_feature(self, field_name: str, value):
        return self.schema.encode_feature(field_name, value)

    def encode_dict(self, values: dict) -> dict:
        return self.schema.encode_dict(values)

    # ---------- 推理 ----------

    def predict_encoded(self, encoded: dict) -> np.ndarray:
        data = self.schema.vector_from_dict(encoded)
        return self.model.predict(data)

    def predict_raw(self, values: dict) -> np.ndarray:
        encoded = self.encode_dict(values)
        return self.predict_encoded(encoded)

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        data = self.schema.vector_from_dataframe(df)
        return self.model.predict(data)

    def predict_from_pydantic(self, data) -> np.ndarray:
        encoded = {f: self.encode_feature(f, getattr(data, f)) for f in self.feature_order}
        return self.predict_encoded(encoded)

    # ---------- 导出 bundle ----------

    def to_bundle(self) -> dict:
        return bundle_model(self.model, self.schema)
