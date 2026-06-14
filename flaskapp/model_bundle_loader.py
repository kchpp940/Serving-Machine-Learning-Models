import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model_runtime import CarPriceModel  # noqa: E402

MODEL_BUNDLE_PATH = os.path.join(_HERE, "models", "sklearn_gbr.pkl")

_MODEL_SINGLETON: CarPriceModel | None = None


def get_model() -> CarPriceModel:
    """返回全局单例的 CarPriceModel 适配器（兼容 bundle / 旧裸模型）。"""
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        _MODEL_SINGLETON = CarPriceModel.from_joblib(MODEL_BUNDLE_PATH)
    return _MODEL_SINGLETON


# ---------------------------------------------------------------------------
# 兼容旧接口：原来的代码使用 get_bundle() 返回 bundle 对象，字段结构略有不同。
# 这里提供一个 "bundle 风格" 的包装，避免 app.py / templates 大改。
# ---------------------------------------------------------------------------

class _BundleStyleWrapper:
    def __init__(self, model: CarPriceModel):
        self._model = model
        self.model = model.model
        self.feature_order = list(model.feature_order)
        self.categorical_encoders = dict(model.categorical_encoders)
        self.target_column = model.target_column

        self.fields = self._build_fields()
        self.template_field_order = ["names"] + list(model.feature_order)

    FIELD_META = {
        "names": {
            "type": "text",
            "label": "Name of Car",
            "placeholder": "Enter Name of Car (e.g. Toyota Camry)",
            "is_model_feature": False,
        },
        "enginesize": {
            "type": "numeric", "label": "Engine Size",
            "placeholder": "Engine displacement in cubic inches (e.g. 130)",
        },
        "curbweight": {
            "type": "numeric", "label": "Curb Weight",
            "placeholder": "Weight in pounds (e.g. 2548)",
        },
        "horsepower": {
            "type": "numeric", "label": "Horse Power",
            "placeholder": "Horsepower output (e.g. 111)",
        },
        "highwaympg": {
            "type": "numeric", "label": "Highway Miles Per Gallon",
            "placeholder": "Highway fuel efficiency (e.g. 27)",
        },
        "carwidth": {
            "type": "numeric", "label": "Car Width",
            "placeholder": "Width in inches (e.g. 64.1)",
        },
        "wheelbase": {
            "type": "numeric", "label": "Wheel Base",
            "placeholder": "Distance between front and rear axles in inches (e.g. 88.6)",
        },
        "drivewheel": {
            "type": "categorical", "label": "Drive Wheel",
            "placeholder": "Select drive wheel configuration",
        },
        "citympg": {
            "type": "numeric", "label": "City Miles Per Gallon",
            "placeholder": "City fuel efficiency (e.g. 21)",
        },
        "boreratio": {
            "type": "numeric", "label": "Bore Ratio",
            "placeholder": "Engine bore ratio (e.g. 3.47)",
        },
        "cylindernumber": {
            "type": "categorical", "label": "Number of Cylinders",
            "placeholder": "Select number of engine cylinders",
        },
    }

    def _build_fields(self) -> dict:
        fields = {"names": dict(self.FIELD_META["names"])}
        for col in self._model.feature_order:
            base = self.FIELD_META.get(col, {})
            if col in self._model.categorical_encoders:
                fields[col] = {
                    **base,
                    "type": "categorical",
                    "options": self._model.categorical_options(col),
                }
            else:
                fields[col] = {**base, "type": "numeric"}
            fields[col]["label"] = base.get("label", col)
            fields[col]["placeholder"] = base.get("placeholder", col)
            fields[col]["is_model_feature"] = True
        return fields

    def encode_form_value(self, field_name: str, form_value: str) -> float:
        return self._model.encode_feature(field_name, form_value)

    def valid_form_values(self, field_name: str) -> set:
        if field_name in self.categorical_encoders:
            return set(self.categorical_encoders[field_name].classes_)
        return set()

    def predict(self, features: dict):
        return self._model.predict_encoded(features)


_BUNDLE_WRAPPER: _BundleStyleWrapper | None = None


def get_bundle() -> _BundleStyleWrapper:
    global _BUNDLE_WRAPPER
    if _BUNDLE_WRAPPER is None:
        _BUNDLE_WRAPPER = _BundleStyleWrapper(get_model())
    return _BUNDLE_WRAPPER
