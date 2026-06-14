from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd

try:
    import joblib as _joblib
except ImportError:  # pragma: no cover
    _joblib = None

try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:  # pragma: no cover
    LabelEncoder = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SHARED_MODEL_DIR = os.path.join(PROJECT_ROOT, "shared_models")
SHARED_MODEL_PATH = os.path.join(SHARED_MODEL_DIR, "sklearn_gbr.pkl")

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
CARS_CSV_PATH = os.path.join(DATA_DIR, "cars.csv")

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

DRIVEWHEEL_DISPLAY: Dict[str, str] = {
    "4wd": "Four Wheel Drive (4WD)",
    "fwd": "Front Wheel Drive (FWD)",
    "rwd": "Rear Wheel Drive (RWD)",
}

FIELD_LABELS: Dict[str, str] = {
    "names": "Car Name",
    "enginesize": "Engine Size",
    "curbweight": "Curb Weight",
    "horsepower": "Horsepower",
    "highwaympg": "Highway MPG",
    "carwidth": "Car Width",
    "wheelbase": "Wheelbase",
    "drivewheel": "Drive Wheel",
    "citympg": "City MPG",
    "boreratio": "Bore Ratio",
    "cylindernumber": "Number of Cylinders",
}

FIELD_DESCRIPTIONS: Dict[str, str] = {
    "enginesize": "发动机排量 (立方英寸)",
    "curbweight": "整备质量 (磅)",
    "horsepower": "最大马力",
    "highwaympg": "高速路百公里油耗换算 (MPG)",
    "carwidth": "车身宽度 (英寸)",
    "wheelbase": "轴距 (英寸)",
    "drivewheel": "驱动轮类型: fwd/4wd/rwd",
    "citympg": "城市百公里油耗换算 (MPG)",
    "boreratio": "气缸内径与冲程比值",
    "cylindernumber": "气缸数: two/four/six/eight 等",
}

FIELD_PLACEHOLDERS: Dict[str, str] = {
    "enginesize": "e.g. 130",
    "curbweight": "e.g. 2548",
    "horsepower": "e.g. 111",
    "highwaympg": "e.g. 27",
    "carwidth": "e.g. 64.1",
    "wheelbase": "e.g. 88.6",
    "citympg": "e.g. 21",
    "boreratio": "e.g. 3.47",
}

DEFAULT_EXAMPLE: Dict = {
    "enginesize": 130,
    "curbweight": 2548,
    "horsepower": 111,
    "highwaympg": 27,
    "carwidth": 64.1,
    "wheelbase": 88.6,
    "drivewheel": "rwd",
    "citympg": 21,
    "boreratio": 3.47,
    "cylindernumber": "four",
}


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

    def vector_from_dict(self, values: dict) -> pd.DataFrame:
        encoded = self.encode_dict(values)
        vector = [[encoded[name] for name in self.feature_order]]
        return pd.DataFrame(vector, columns=self.feature_order)

    def vector_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded_rows = []
        for _, row in df.iterrows():
            encoded_rows.append(
                [self.encode_feature(f, row[f]) for f in self.feature_order]
            )
        return pd.DataFrame(encoded_rows, columns=self.feature_order)

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
                "form_value": str(raw),
                "model_code": int(code),
            })
        return options

    def to_dict(self) -> dict:
        encoder_data = {}
        for col, le in self.categorical_encoders.items():
            encoder_data[col] = {
                "classes": [str(c) for c in le.classes_],
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
            raise SchemaMismatchError(
                "model_n_features",
                f"模型 n_features_in_={n_features} 与 schema 特征数 {self.n_features()} 不一致",
            )

    def validate_against_interface(self, interface_fields: Sequence[str]) -> None:
        schema_set = set(self.feature_order)
        interface_set = set(interface_fields)
        missing = schema_set - interface_set
        extra = interface_set - schema_set
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"接口缺少字段: {sorted(missing)}")
            if extra:
                parts.append(f"接口多余字段: {sorted(extra)}")
            raise SchemaMismatchError(
                "interface_mismatch",
                f"服务接口字段与 schema 不一致 — {'; '.join(parts)}",
            )

    def default_example(self) -> dict:
        """返回合法的请求示例，字段顺序按 feature_order。"""
        return {name: DEFAULT_EXAMPLE[name] for name in self.feature_order}

    def field_label(self, field_name: str) -> str:
        return FIELD_LABELS.get(field_name, field_name)

    def field_description(self, field_name: str) -> str:
        return FIELD_DESCRIPTIONS.get(field_name, "")

    def field_type(self, field_name: str) -> str:
        if field_name in self.numeric_features:
            return "number"
        if field_name in self.categorical_features:
            return "string"
        return "unknown"

    def field_placeholder(self, field_name: str) -> str:
        if field_name in self.numeric_features:
            return FIELD_PLACEHOLDERS.get(
                field_name,
                f"e.g. {DEFAULT_EXAMPLE.get(field_name, '')}",
            )
        return ""

    def field_allowed_values(self, field_name: str) -> list:
        """返回字段合法输入值列表；数值字段返回空列表。"""
        if field_name not in self.feature_order:
            raise ValueError(f"未知字段: {field_name}")
        if field_name in self.categorical_features:
            return [str(c) for c in self.categorical_classes(field_name)]
        return []

    def field_error_message(self, field_name: str, error_type: str) -> str:
        """统一错误提示模板。error_type: required/type/invalid."""
        label = self.field_label(field_name)
        if error_type == "required":
            return f"{label} cannot be empty."
        if error_type == "type":
            if field_name in self.numeric_features:
                return f"{label} must be a valid number (decimals allowed)."
            return f"{label} must be a valid string."
        if error_type == "invalid":
            if field_name in self.categorical_features:
                allowed = ", ".join(self.field_allowed_values(field_name))
                return f"{label} must be one of: [{allowed}]"
            return f"{label} contains an invalid value."
        return f"{label} has an error."

    def schema_export(self) -> dict:
        """FastAPI /schema 端点标准返回体。三端共用。"""
        return {
            "feature_order": list(self.feature_order),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "target_column": self.target_column,
            "fields": {
                name: {
                    "name": name,
                    "label": self.field_label(name),
                    "description": self.field_description(name),
                    "type": self.field_type(name),
                    "python_type": "float" if self.field_type(name) == "number" else "str",
                    "placeholder": self.field_placeholder(name),
                    "allowed_values": self.field_allowed_values(name),
                    "options": (
                        self.categorical_options(name)
                        if name in self.categorical_features
                        else []
                    ),
                    "example": DEFAULT_EXAMPLE.get(name),
                    "error_messages": {
                        "required": self.field_error_message(name, "required"),
                        "type": self.field_error_message(name, "type"),
                        "invalid": self.field_error_message(name, "invalid"),
                    },
                }
                for name in self.feature_order
            },
            "example": self.default_example(),
        }

    def form_fields_metadata(self, extra_fields: list = None) -> list:
        """Flask 模板使用的表单元数据列表，按 feature_order 顺序。

        extra_fields: 额外插入到表单开头的非模型字段 (如 "names")。
        """
        items = []
        if extra_fields:
            for name in extra_fields:
                items.append({
                    "name": name,
                    "label": FIELD_LABELS.get(name, name),
                    "description": FIELD_DESCRIPTIONS.get(name, ""),
                    "kind": "text_extra",
                    "placeholder": "",
                    "allowed_values": [],
                    "options": [],
                    "example": "",
                    "required_error": f"{FIELD_LABELS.get(name, name)} cannot be empty.",
                    "type_error": "",
                })
        for name in self.feature_order:
            if name in self.categorical_features:
                kind = "select"
            else:
                kind = "number"
            items.append({
                "name": name,
                "label": self.field_label(name),
                "description": self.field_description(name),
                "kind": kind,
                "placeholder": self.field_placeholder(name),
                "allowed_values": self.field_allowed_values(name),
                "options": (
                    self.categorical_options(name)
                    if name in self.categorical_features
                    else []
                ),
                "example": DEFAULT_EXAMPLE.get(name),
                "required_error": self.field_error_message(name, "required"),
                "type_error": self.field_error_message(name, "type"),
                "invalid_error": self.field_error_message(name, "invalid"),
            })
        return items

    def input_spec(self) -> dict:
        """BentoML 或 OpenAPI 使用的输入规范描述。"""
        return {
            "format": "dataframe or dict",
            "n_features": self.n_features(),
            "columns": list(self.feature_order),
            "column_dtypes": {
                name: ("float32" if self.field_type(name) == "number" else "string")
                for name in self.feature_order
            },
            "example": self.default_example(),
            "fields": [
                {
                    "name": name,
                    "type": self.field_type(name),
                    "allowed_values": self.field_allowed_values(name),
                }
                for name in self.feature_order
            ],
        }


class SchemaMismatchError(RuntimeError):
    def __init__(self, check_name: str, detail: str):
        self.check_name = check_name
        self.detail = detail
        super().__init__(f"[{check_name}] {detail}")


def validate_service_schema(
    schema: FeatureSchema,
    model,
    interface_fields: Sequence[str],
) -> List[str]:
    errors: List[str] = []

    try:
        schema.validate()
    except (ValueError, SchemaMismatchError) as e:
        errors.append(str(e))

    try:
        schema.validate_model_input(model)
    except SchemaMismatchError as e:
        errors.append(str(e))

    try:
        schema.validate_against_interface(interface_fields)
    except SchemaMismatchError as e:
        errors.append(str(e))

    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and len(interface_fields) != n_features:
        errors.append(
            f"[model_vs_interface] 模型 n_features_in_={n_features} "
            f"但接口字段数={len(interface_fields)}"
        )

    return errors


def _display_name(field_name: str, raw_class: str) -> str:
    if field_name == "drivewheel":
        return DRIVEWHEEL_DISPLAY.get(raw_class, raw_class.upper())
    if field_name == "cylindernumber":
        num = WORD_TO_NUM_CYLINDERS.get(raw_class.lower())
        if num is not None:
            return f"{num} cylinders"
        return raw_class
    return raw_class


def load_training_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = CARS_CSV_PATH
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
    return {
        "model": model,
        "schema": schema.to_dict(),
    }


def is_model_bundle(obj) -> bool:
    return isinstance(obj, dict) and "model" in obj and "schema" in obj


def save_bundle(bundle: dict, path: str) -> None:
    if _joblib is None:
        raise RuntimeError("需要 joblib 才能保存模型")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _joblib.dump(bundle, path)


def load_bundle(path: str) -> dict:
    if _joblib is None:
        raise RuntimeError("需要 joblib 才能加载模型")
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    return _joblib.load(path)


def find_model_path(local_dir: str | None = None, local_name: str = "sklearn_gbr.pkl") -> str:
    if local_dir is not None:
        local_path = os.path.join(local_dir, local_name)
        if os.path.exists(local_path):
            return local_path
    if os.path.exists(SHARED_MODEL_PATH):
        return SHARED_MODEL_PATH
    raise FileNotFoundError(
        f"未找到模型文件: 已搜索 {local_dir or '<无本地路径>'} 和 {SHARED_MODEL_PATH}"
    )


# -----------------------------
# Pydantic v1 / v2 兼容辅助
# -----------------------------

try:
    import pydantic as _pydantic
    _PYDANTIC_MAJOR = int(_pydantic.VERSION.split(".")[0])
except ImportError:  # pragma: no cover
    _pydantic = None
    _PYDANTIC_MAJOR = 0


def pydantic_major_version() -> int:
    """返回当前安装的 Pydantic 主版本号 (1 或 2)；未安装时返回 0。"""
    return _PYDANTIC_MAJOR


def pydantic_field_names(model_cls) -> List[str]:
    """跨 v1/v2 获取模型字段名列表，按定义顺序。"""
    if _PYDANTIC_MAJOR >= 2:
        return list(model_cls.model_fields.keys())
    return list(model_cls.__fields__.keys())


def build_car_prediction_model(
    feature_order: List[str] = None,
    categorical_features: List[str] = None,
    example: dict = None,
    class_name: str = "CarPrediction",
):
    """根据 FeatureSchema 常量动态生成 Pydantic 请求模型，v1/v2 都可用。

    - 数值字段 -> float
    - 分类字段 -> str
    - 自动校验生成的模型字段与 feature_order 完全一致
    - 通过 `pydantic_field_names()` 读取字段顺序，避免 v1/v2 API 漂移
    """
    if _pydantic is None:
        raise RuntimeError("需要 pydantic 才能使用 build_car_prediction_model")

    if feature_order is None:
        feature_order = list(FEATURE_ORDER)
    if categorical_features is None:
        categorical_features = list(CATEGORICAL_FEATURES)
    if example is None:
        example = dict(DEFAULT_EXAMPLE)

    cat_set = set(categorical_features)
    annotations = {}
    for name in feature_order:
        annotations[name] = str if name in cat_set else float

    namespace = {"__annotations__": annotations, "__module__": __name__}

    if _PYDANTIC_MAJOR >= 2:
        namespace["model_config"] = {"json_schema_extra": {"example": example}}
    else:
        config_ns = {"schema_extra": {"example": example}}
        namespace["Config"] = type("Config", (), config_ns)

    BaseModel = _pydantic.BaseModel
    model_cls = type(class_name, (BaseModel,), namespace)

    actual = pydantic_field_names(model_cls)
    if actual != list(feature_order):
        raise RuntimeError(
            f"生成的 Pydantic 模型字段 {actual} 与 feature_order {list(feature_order)} 不一致"
        )

    return model_cls


def interface_fields_from_model(model_cls) -> List[str]:
    """从 Pydantic 模型提取接口字段列表，保证三端校验使用同一份字段定义。"""
    return pydantic_field_names(model_cls)
