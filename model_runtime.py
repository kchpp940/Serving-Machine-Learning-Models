from __future__ import annotations

import os
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover - 运行环境必定有 joblib
    joblib = None

_DEFAULT_FEATURE_ORDER_LEGACY = [
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

_NUMERIC_FEATURE_SET = {
    "enginesize", "curbweight", "horsepower", "highwaympg",
    "carwidth", "wheelbase", "citympg", "boreratio",
}

_CATEGORICAL_FEATURE_SET = {"drivewheel", "cylindernumber"}

_WORD_TO_NUM_CYLINDERS = {
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}

_NUM_WORDS_CYLINDERS = {v: k for k, v in _WORD_TO_NUM_CYLINDERS.items()}


def _is_string_dtype(dtype) -> bool:
    return pd.api.types.is_string_dtype(dtype) or dtype == "O"


def _is_bundle(obj) -> bool:
    return isinstance(obj, dict) and "model" in obj and "feature_order" in obj


def _display_name(field_name: str, raw_class: str) -> str:
    if field_name == "drivewheel":
        mapping = {
            "4wd": "Four Wheel Drive (4WD)",
            "fwd": "Front Wheel Drive (FWD)",
            "rwd": "Rear Wheel Drive (RWD)",
        }
        return mapping.get(raw_class, raw_class.upper())
    if field_name == "cylindernumber":
        num = _WORD_TO_NUM_CYLINDERS.get(raw_class.lower())
        if num is not None:
            return f"{num} cylinders"
        return raw_class
    return raw_class


class CarPriceModel:
    """统一的推理适配器：兼容 bundle dict 和旧版裸模型两种 pkl 格式。

    - Bundle 模式 (推荐)：从 categorical_encoders 读取真实 LabelEncoder，
      按 feature_order 组装向量。
    - Legacy 模式（旧裸模型文件）：回退到硬编码的 10 特征顺序和
      LabelEncoder 字母序映射，保证线上旧模型不会崩。
    """

    def __init__(self, raw):
        self._raw = raw
        if _is_bundle(raw):
            self.mode = "bundle"
            self.model = raw["model"]
            self.feature_order = list(raw["feature_order"])
            self.categorical_encoders = dict(raw.get("categorical_encoders", {}))
            self.target_column = raw.get("target_column", "price")
        else:
            self.mode = "legacy"
            self.model = raw
            self.feature_order = list(_DEFAULT_FEATURE_ORDER_LEGACY)
            self.categorical_encoders = self._build_legacy_encoders()
            self.target_column = "price"

        self._validate_dimensions()

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
        expected = len(self.feature_order)
        actual = getattr(self.model, "n_features_in_", expected)
        if actual != expected:
            raise RuntimeError(
                f"模型 n_features_in_={actual} 与 feature_order 长度 {expected} 不一致"
            )
        for col in self.categorical_encoders:
            if col not in self.feature_order:
                raise RuntimeError(
                    f"分类编码器包含未在 feature_order 中出现的列: {col}"
                )

    @classmethod
    def from_joblib(cls, path: str) -> "CarPriceModel":
        if joblib is None:
            raise RuntimeError("需要先安装 joblib")
        raw = joblib.load(path)
        return cls(raw)

    @classmethod
    def from_sklearn_object(cls, model) -> "CarPriceModel":
        """BentoML 或直接把 sklearn 对象传入时使用。"""
        if _is_bundle(model):
            return cls(model)
        return cls(model)

    # ---------- 元数据 ----------

    @property
    def numeric_features(self) -> list:
        return [f for f in self.feature_order if f in _NUMERIC_FEATURE_SET]

    @property
    def categorical_features(self) -> list:
        return [f for f in self.feature_order if f in _CATEGORICAL_FEATURE_SET]

    def categorical_classes(self, field_name: str):
        lb = self.categorical_encoders.get(field_name)
        if lb is None:
            raise ValueError(f"{field_name} 不是分类字段或无编码器")
        return list(lb.classes_)

    def categorical_options(self, field_name: str) -> list:
        lb = self.categorical_encoders.get(field_name)
        if lb is None:
            raise ValueError(f"{field_name} 不是分类字段或无编码器")
        codes = lb.transform(lb.classes_)
        options = []
        for raw, code in zip(lb.classes_, codes):
            options.append({
                "display": _display_name(field_name, raw),
                "form_value": raw,
                "model_code": int(code),
            })
        return options

    # ---------- 编码 ----------

    def encode_feature(self, field_name: str, value):
        if field_name not in self.feature_order:
            raise ValueError(f"未知字段: {field_name}")
        if field_name in self.categorical_encoders:
            lb = self.categorical_encoders[field_name]
            valid_codes = set(range(len(lb.classes_)))

            if isinstance(value, bool):
                s = str(value).lower()
            elif isinstance(value, (int, float)):
                if isinstance(value, float) and not value.is_integer():
                    valid_words = ", ".join(sorted(lb.classes_))
                    raise ValueError(
                        f"{field_name} 的合法值为 [{valid_words}] 或整数编码 "
                        f"{sorted(valid_codes)}, 收到 {value!r}"
                    )
                code = int(value)
                if code not in valid_codes:
                    valid_words = ", ".join(sorted(lb.classes_))
                    raise ValueError(
                        f"{field_name} 的合法值为 [{valid_words}] 或整数编码 "
                        f"{sorted(valid_codes)}, 收到 {value!r}"
                    )
                return float(code)
            else:
                s = str(value).strip()
                if s in set(lb.classes_):
                    return float(lb.transform([s])[0])
                # 兼容"字符串形式的数字编码"，比如用户写 "2"
                try:
                    code = int(s)
                except ValueError:
                    code = None
                if code is not None and code in valid_codes:
                    return float(code)
                valid_words = ", ".join(sorted(lb.classes_))
                raise ValueError(
                    f"{field_name} 的合法值为 [{valid_words}] 或整数编码 "
                    f"{sorted(valid_codes)}, 收到 {value!r}"
                )
        return float(value)

    def encode_dict(self, values: dict) -> dict:
        return {f: self.encode_feature(f, values[f]) for f in self.feature_order}

    # ---------- 推理 ----------

    def _vector_from_encoded(self, encoded: dict) -> np.ndarray:
        vector = [encoded[name] for name in self.feature_order]
        return np.array([vector])

    def predict_encoded(self, encoded: dict) -> np.ndarray:
        data = self._vector_from_encoded(encoded)
        return self.model.predict(data)

    def predict_raw(self, values: dict) -> np.ndarray:
        encoded = self.encode_dict(values)
        return self.predict_encoded(encoded)

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        encoded_rows = []
        for _, row in df.iterrows():
            encoded_rows.append(
                [self.encode_feature(f, row[f]) for f in self.feature_order]
            )
        data = np.array(encoded_rows)
        return self.model.predict(data)

    # ---------- FastAPI 兼容层 ----------

    def predict_from_pydantic(self, data) -> np.ndarray:
        """接受 pydantic CarPrediction（字段名对齐即可，分类字段允许字符串或数字编码）。"""
        encoded = {f: self.encode_feature(f, getattr(data, f)) for f in self.feature_order}
        return self.predict_encoded(encoded)
