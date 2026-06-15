from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any


PREDICTION_CURRENCY: str = "USD"


FIELD_PREDICTION: str = "prediction"
FIELD_CURRENCY: str = "currency"
FIELD_MODEL_NAME: str = "model_name"
FIELD_TOP_FEATURES: str = "top_features"
FIELD_FEATURE_VALUES: str = "feature_values"
FIELD_FEATURE: str = "feature"
FIELD_LABEL: str = "label"
FIELD_VALUE: str = "value"
FIELD_DISPLAY: str = "display"
FIELD_GLOBAL_IMPORTANCE: str = "global_importance"
FIELD_GLOBAL_IMPORTANCE_PERCENT: str = "global_importance_percent"


GLOBAL_IMPORTANCE_DESCRIPTION: str = (
    "Model-level feature importance score (normalized, sum = 1). "
    "This is a global property of the trained model, not a contribution specific to this input."
)

GLOBAL_IMPORTANCE_PERCENT_DESCRIPTION: str = (
    "Model-level feature importance expressed as a percentage (0-100)."
)

EXPLAIN_TOP_FEATURES_DESCRIPTION: str = (
    "Top N features ranked by model-level (global) feature importance. "
    "These are global importance scores from the trained model, not per-sample SHAP-style contributions."
)

EXPLAIN_FEATURE_VALUES_DESCRIPTION: str = (
    "Input feature values keyed by internal feature code. "
    "Each entry carries the raw value, the feature display label, and a human-readable value string."
)

PREDICTION_RESPONSE_FIELDS = (FIELD_PREDICTION, FIELD_CURRENCY, FIELD_MODEL_NAME)
EXPLAIN_RESPONSE_FIELDS = (
    FIELD_PREDICTION, FIELD_CURRENCY, FIELD_MODEL_NAME,
    FIELD_TOP_FEATURES, FIELD_FEATURE_VALUES,
)
GLOBAL_FEATURE_IMPORTANCE_FIELDS = (
    FIELD_FEATURE, FIELD_LABEL,
    FIELD_GLOBAL_IMPORTANCE, FIELD_GLOBAL_IMPORTANCE_PERCENT,
)
INPUT_FEATURE_VALUE_FIELDS = (FIELD_VALUE, FIELD_LABEL, FIELD_DISPLAY)


@dataclass
class GlobalFeatureImportanceItem:
    feature: str
    label: str
    global_importance: float
    global_importance_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            FIELD_FEATURE: self.feature,
            FIELD_LABEL: self.label,
            FIELD_GLOBAL_IMPORTANCE: self.global_importance,
            FIELD_GLOBAL_IMPORTANCE_PERCENT: self.global_importance_percent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlobalFeatureImportanceItem":
        return cls(
            feature=data[FIELD_FEATURE],
            label=data[FIELD_LABEL],
            global_importance=float(data[FIELD_GLOBAL_IMPORTANCE]),
            global_importance_percent=float(data[FIELD_GLOBAL_IMPORTANCE_PERCENT]),
        )


@dataclass
class InputFeatureValueItem:
    value: Any
    label: str
    display: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            FIELD_VALUE: self.value,
            FIELD_LABEL: self.label,
            FIELD_DISPLAY: self.display,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputFeatureValueItem":
        return cls(
            value=data[FIELD_VALUE],
            label=data[FIELD_LABEL],
            display=str(data[FIELD_DISPLAY]),
        )


@dataclass
class PredictionResult:
    prediction: float
    currency: str = PREDICTION_CURRENCY
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            FIELD_PREDICTION: self.prediction,
            FIELD_CURRENCY: self.currency,
            FIELD_MODEL_NAME: self.model_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionResult":
        return cls(
            prediction=float(data[FIELD_PREDICTION]),
            currency=str(data.get(FIELD_CURRENCY, PREDICTION_CURRENCY)),
            model_name=str(data.get(FIELD_MODEL_NAME, "")),
        )


@dataclass
class ExplainResult:
    prediction: float
    currency: str = PREDICTION_CURRENCY
    model_name: str = ""
    top_features: List[GlobalFeatureImportanceItem] = field(default_factory=list)
    feature_values: Dict[str, InputFeatureValueItem] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            FIELD_PREDICTION: self.prediction,
            FIELD_CURRENCY: self.currency,
            FIELD_MODEL_NAME: self.model_name,
            FIELD_TOP_FEATURES: [item.to_dict() for item in self.top_features],
            FIELD_FEATURE_VALUES: {k: v.to_dict() for k, v in self.feature_values.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplainResult":
        return cls(
            prediction=float(data[FIELD_PREDICTION]),
            currency=str(data.get(FIELD_CURRENCY, PREDICTION_CURRENCY)),
            model_name=str(data.get(FIELD_MODEL_NAME, "")),
            top_features=[
                GlobalFeatureImportanceItem.from_dict(item)
                for item in data.get(FIELD_TOP_FEATURES, [])
            ],
            feature_values={
                k: InputFeatureValueItem.from_dict(v)
                for k, v in data.get(FIELD_FEATURE_VALUES, {}).items()
            },
        )


def build_global_feature_importance_item(
    feature: str,
    label: str,
    global_importance: float,
) -> GlobalFeatureImportanceItem:
    return GlobalFeatureImportanceItem(
        feature=feature,
        label=label,
        global_importance=global_importance,
        global_importance_percent=round(global_importance * 100, 2),
    )


def build_input_feature_value_item(
    value: Any,
    label: str,
    display: str,
) -> InputFeatureValueItem:
    return InputFeatureValueItem(
        value=value,
        label=label,
        display=display,
    )


def build_explain_result(
    prediction: float,
    model_name: str,
    feature_order: List[str],
    importances: Dict[str, float],
    values: Dict[str, Any],
    label_fn,
    display_fn,
    top_k: int = 5,
    currency: str = PREDICTION_CURRENCY,
) -> ExplainResult:
    all_items = []
    for feature in feature_order:
        item = build_global_feature_importance_item(
            feature=feature,
            label=label_fn(feature),
            global_importance=importances[feature],
        )
        all_items.append(item)

    all_items.sort(key=lambda x: x.global_importance, reverse=True)
    top_items = all_items[:top_k] if top_k and top_k > 0 else all_items

    feature_values = {}
    for feature in feature_order:
        raw_value = values.get(feature)
        feature_values[feature] = build_input_feature_value_item(
            value=raw_value,
            label=label_fn(feature),
            display=display_fn(feature, raw_value),
        )

    return ExplainResult(
        prediction=prediction,
        currency=currency,
        model_name=model_name,
        top_features=top_items,
        feature_values=feature_values,
    )


def build_prediction_result(
    prediction: float,
    model_name: str,
    currency: str = PREDICTION_CURRENCY,
) -> PredictionResult:
    return PredictionResult(
        prediction=prediction,
        currency=currency,
        model_name=model_name,
    )
