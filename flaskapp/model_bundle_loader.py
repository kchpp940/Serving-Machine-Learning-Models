import os
import joblib
import numpy as np

MODEL_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "models", "sklearn_gbr.pkl")

FIELD_META = {
    "names": {
        "type": "text",
        "label": "Name of Car",
        "placeholder": "Enter Name of Car (e.g. Toyota Camry)",
        "is_model_feature": False,
    },
    "enginesize": {
        "type": "numeric",
        "label": "Engine Size",
        "placeholder": "Engine displacement in cubic inches (e.g. 130)",
    },
    "curbweight": {
        "type": "numeric",
        "label": "Curb Weight",
        "placeholder": "Weight in pounds (e.g. 2548)",
    },
    "horsepower": {
        "type": "numeric",
        "label": "Horse Power",
        "placeholder": "Horsepower output (e.g. 111)",
    },
    "highwaympg": {
        "type": "numeric",
        "label": "Highway Miles Per Gallon",
        "placeholder": "Highway fuel efficiency (e.g. 27)",
    },
    "carwidth": {
        "type": "numeric",
        "label": "Car Width",
        "placeholder": "Width in inches (e.g. 64.1)",
    },
    "wheelbase": {
        "type": "numeric",
        "label": "Wheel Base",
        "placeholder": "Distance between front and rear axles in inches (e.g. 88.6)",
    },
    "drivewheel": {
        "type": "categorical",
        "label": "Drive Wheel",
        "placeholder": "Select drive wheel configuration",
    },
    "citympg": {
        "type": "numeric",
        "label": "City Miles Per Gallon",
        "placeholder": "City fuel efficiency (e.g. 21)",
    },
    "boreratio": {
        "type": "numeric",
        "label": "Bore Ratio",
        "placeholder": "Engine bore ratio (e.g. 3.47)",
    },
    "cylindernumber": {
        "type": "categorical",
        "label": "Number of Cylinders",
        "placeholder": "Select number of engine cylinders",
    },
}

_NUM_WORDS = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "11",
    12: "12",
}

_WORD_TO_NUM = {
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}


def _make_display_name(field_name: str, raw_class: str) -> str:
    if field_name == "drivewheel":
        mapping = {"4wd": "Four Wheel Drive (4WD)", "fwd": "Front Wheel Drive (FWD)", "rwd": "Rear Wheel Drive (RWD)"}
        return mapping.get(raw_class, raw_class.upper())
    if field_name == "cylindernumber":
        num = _WORD_TO_NUM.get(raw_class.lower())
        if num is not None:
            return f"{num} cylinders"
        return raw_class
    return raw_class


class ModelBundle:
    def __init__(self, bundle_path: str = MODEL_BUNDLE_PATH):
        raw = joblib.load(bundle_path)
        if not isinstance(raw, dict) or "model" not in raw or "feature_order" not in raw:
            raise RuntimeError(
                f"{bundle_path} 不是合法的模型包；请先用训练脚本生成包含 model/feature_order/categorical_encoders 的 bundle"
            )

        self._raw = raw
        self.model = raw["model"]
        self.feature_order = list(raw["feature_order"])
        self.categorical_encoders = dict(raw.get("categorical_encoders", {}))
        self.target_column = raw.get("target_column", "price")

        n_expected = len(self.feature_order)
        n_actual = getattr(self.model, "n_features_in_", n_expected)
        if n_actual != n_expected:
            raise RuntimeError(
                f"模型 n_features_in_={n_actual} 与 feature_order 长度 {n_expected} 不一致"
            )

        for col in self.categorical_encoders:
            if col not in self.feature_order:
                raise RuntimeError(
                    f"分类编码器包含未在 feature_order 中出现的列: {col}"
                )

        self.fields = self._build_fields()
        self.template_field_order = ["names"] + list(self.feature_order)

    def _build_fields(self) -> dict:
        fields = {"names": dict(FIELD_META["names"])}
        for col in self.feature_order:
            base = FIELD_META.get(col, {})
            if col in self.categorical_encoders:
                lb = self.categorical_encoders[col]
                codes = lb.transform(lb.classes_)
                options = []
                for raw_class, code in zip(lb.classes_, codes):
                    options.append({
                        "display": _make_display_name(col, raw_class),
                        "form_value": raw_class,
                        "model_code": int(code),
                    })
                fields[col] = {
                    **base,
                    "type": "categorical",
                    "options": options,
                }
            else:
                fields[col] = {**base, "type": "numeric"}
            fields[col]["label"] = base.get("label", col)
            fields[col]["placeholder"] = base.get("placeholder", col)
            fields[col]["is_model_feature"] = True
        return fields

    def encode_form_value(self, field_name: str, form_value: str) -> float:
        if field_name not in self.feature_order:
            raise ValueError(f"未知字段: {field_name}")
        if field_name in self.categorical_encoders:
            lb = self.categorical_encoders[field_name]
            valid = set(lb.classes_)
            if form_value not in valid:
                raise ValueError(
                    f"{field_name} 的合法值为 {sorted(valid)}, 收到 {form_value!r}"
                )
            return float(lb.transform([form_value])[0])
        return float(form_value)

    def valid_form_values(self, field_name: str) -> set:
        if field_name in self.categorical_encoders:
            return set(self.categorical_encoders[field_name].classes_)
        return set()

    def predict(self, features: dict) -> np.ndarray:
        vector = [features[name] for name in self.feature_order]
        data = np.array([vector])
        return self.model.predict(data)


_BUNDLE_SINGLETON: ModelBundle | None = None


def get_bundle() -> ModelBundle:
    global _BUNDLE_SINGLETON
    if _BUNDLE_SINGLETON is None:
        _BUNDLE_SINGLETON = ModelBundle()
    return _BUNDLE_SINGLETON
