from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:  # pragma: no cover
    LabelEncoder = None


FEATURE_ORDER: List[str] = [
    "enginesize",
    "curbweight",
    "horsepower",
    "highwaympg",
    "carwidth",
    "wheelbase",
    "drivewheel",
    "citympg",
    "boreratio",
    "cylindernumber",
]

NUMERIC_FEATURES: List[str] = [
    "enginesize",
    "curbweight",
    "horsepower",
    "highwaympg",
    "carwidth",
    "wheelbase",
    "citympg",
    "boreratio",
]

CATEGORICAL_FEATURES: List[str] = [
    "drivewheel",
    "cylindernumber",
]

TARGET_COLUMN: str = "price"

WORD_TO_NUM_CYLINDERS: Dict[str, int] = {
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}

NUM_TO_WORD_CYLINDERS: Dict[int, str] = {v: k for k, v in WORD_TO_NUM_CYLINDERS.items()}

DRIVEWheel_DISPLAY: Dict[str, str] = {
    "4wd": "Four Wheel Drive (4WD)",
    "fwd": "Front Wheel Drive (FWD)",
    "rwd": "Rear Wheel Drive (RWD)",
}


def _is_string_dtype(dtype) -> bool:
    return pd.api.types.is_string_dtype(dtype) or dtype == "O"


@dataclass
class FeatureSchema:
    feature_order: List[str] = field(default_factory=lambda: list(FEATURE_ORDER))
    numeric_features: List[str] = field(default_factory=lambda: list(NUMERIC_FEATURES))
    categorical_features: List[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    target_column: str = TARGET_COLUMN
    categorical_encoders: Dict[str, "LabelEncoder"] = field(default_factory=dict)

    def validate(self) -> None:
        for f in self.numeric_features:
            if f not in self.feature_order:
                raise ValueError(f"数值特征 {f} 不在 feature_order 中")
        for f in self.categorical_features:
            if f not in self.feature_order:
                raise ValueError(f"分类特征 {f} 不在 feature_order 中")
        for f in self.feature_order:
            if f not in self.numeric_features and f not in self.categorical_features:
                raise ValueError(f"特征 {f} 既不是数值也不是分类特征")
        for col in self.categorical_encoders:
            if col not in self.categorical_features:
                raise ValueError(f"编码器 {col} 对应列不是分类特征")

    def n_features(self) -> int:
        return len(self.feature_order)

    def fit_encoders(self, df: pd.DataFrame) -> None:
        if LabelEncoder is None:
            raise RuntimeError("需要 scikit-learn 才能使用 fit_encoders")
        self.categorical_encoders = {}
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.categorical_encoders[col] = le

    def encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df[self.feature_order].copy()
        for col in self.categorical_features:
            if col in self.categorical_encoders:
                le = self.categorical_encoders[col]
                result[col] = le.transform(result[col].astype(str))
            else:
                raise RuntimeError(f"分类列 {col} 没有训练好的编码器")
        return result

    def encode_feature(self, field_name: str, value) -> float:
        if field_name not in self.feature_order:
            raise ValueError(f"未知字段: {field_name}")

        if field_name in self.categorical_features:
            if field_name not in self.categorical_encoders:
                raise RuntimeError(f"分类字段 {field_name} 没有编码器")

            le = self.categorical_encoders[field_name]
            valid_codes = set(range(len(le.classes_)))

            if isinstance(value, bool):
                s = str(value).lower()
            elif isinstance(value, (int, float)):
                if isinstance(value, float) and not value.is_integer():
                    valid_words = ", ".join(sorted(le.classes_))
                    raise ValueError(
                        f"{field_name} 的合法值为 [{valid_words}] 或整数编码 "
                        f"{sorted(valid_codes)}, 收到 {value!r}"
                    )
                code = int(value)
                if code not in valid_codes:
                    valid_words = ", ".join(sorted(le.classes_))
                    raise ValueError(
                        f"{field_name} 的合法值为 [{valid_words}] 或整数编码 "
                        f"{sorted(valid_codes)}, 收到 {value!r}"
                    )
                return float(code)
            else:
                s = str(value).strip()
                if s in set(le.classes_):
                    return float(le.transform([s])[0])
                try:
                    code = int(s)
                except ValueError:
                    code = None
                if code is not None and code in valid_codes:
                    return float(code)
                valid_words = ", ".join(sorted(le.classes_))
                raise ValueError(
                    f"{field_name} 的合法值为 [{valid_words}] 或整数编码 "
                    f"{sorted(valid_codes)}, 收到 {value!r}"
                )

        return float(value)

    def encode_dict(self, values: dict) -> dict:
        return {f: self.encode_feature(f, values[f]) for f in self.feature_order}

    def vector_from_dict(self, values: dict) -> np.ndarray:
        encoded = self.encode_dict(values)
        vector = [encoded[name] for name in self.feature_order]
        return np.array([vector])

    def vector_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        encoded_rows = []
        for _, row in df.iterrows():
            encoded_rows.append(
                [self.encode_feature(f, row[f]) for f in self.feature_order]
            )
        return np.array(encoded_rows)

    def categorical_classes(self, field_name: str) -> list:
        if field_name not in self.categorical_encoders:
            raise ValueError(f"{field_name} 不是分类字段或无编码器")
        return list(self.categorical_encoders[field_name].classes_)

    def categorical_options(self, field_name: str) -> list:
        if field_name not in self.categorical_encoders:
            raise ValueError(f"{field_name} 不是分类字段或无编码器")
        lb = self.categorical_encoders[field_name]
        codes = lb.transform(lb.classes_)
        options = []
        for raw, code in zip(lb.classes_, codes):
            options.append({
                "display": _display_name(field_name, raw),
                "form_value": raw,
                "model_code": int(code),
            })
        return options

    def to_dict(self) -> dict:
        encoder_data = {}
        for col, le in self.categorical_encoders.items():
            encoder_data[col] = {
                "classes": list(le.classes_),
            }
        return {
            "feature_order": list(self.feature_order),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "target_column": self.target_column,
            "categorical_encoders": encoder_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSchema":
        if LabelEncoder is None:
            raise RuntimeError("需要 scikit-learn 才能使用 from_dict")
        schema = cls(
            feature_order=list(data["feature_order"]),
            numeric_features=list(data.get("numeric_features", NUMERIC_FEATURES)),
            categorical_features=list(data.get("categorical_features", CATEGORICAL_FEATURES)),
            target_column=data.get("target_column", TARGET_COLUMN),
        )
        encoders = {}
        for col, enc_data in data.get("categorical_encoders", {}).items():
            le = LabelEncoder()
            le.classes_ = np.array(enc_data["classes"])
            encoders[col] = le
        schema.categorical_encoders = encoders
        return schema

    def validate_model_input(self, model) -> None:
        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None and n_features != self.n_features():
            raise RuntimeError(
                f"模型期望 n_features_in_={n_features} 但 schema 有 {self.n_features()} 个特征"
            )


def _display_name(field_name: str, raw_class: str) -> str:
    if field_name == "drivewheel":
        return DRIVEWheel_DISPLAY.get(raw_class, raw_class.upper())
    if field_name == "cylindernumber":
        num = WORD_TO_NUM_CYLINDERS.get(raw_class.lower())
        if num is not None:
            return f"{num} cylinders"
        return raw_class
    return raw_class


def load_training_data(csv_path: str) -> pd.DataFrame:
    usecols = FEATURE_ORDER + [TARGET_COLUMN]
    df = pd.read_csv(csv_path, usecols=usecols)
    return df


def prepare_training_data(df: pd.DataFrame) -> tuple:
    schema = FeatureSchema()
    schema.fit_encoders(df)
    encoded_df = schema.encode_dataframe(df)
    X = encoded_df[schema.feature_order]
    y = df[TARGET_COLUMN]
    return X, y, schema


def bundle_model(model, schema: FeatureSchema) -> dict:
    bundle = {
        "model": model,
        "schema": schema.to_dict(),
    }
    return bundle


def is_model_bundle(obj) -> bool:
    return isinstance(obj, dict) and "model" in obj and "schema" in obj
