from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


PREDICTION_CURRENCY: str = "USD"

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


@dataclass
class GlobalFeatureImportance:
    feature: str = ""
    label: str = ""
    global_importance: float = 0.0
    global_importance_percent: float = 0.0

    FEATURE_DESCRIPTION: str = field(
        default="Internal feature code name",
        init=False,
        repr=False,
    )
    LABEL_DESCRIPTION: str = field(
        default="Human-readable feature label from the shared feature schema",
        init=False,
        repr=False,
    )
    GLOBAL_IMPORTANCE_DESCRIPTION: str = field(
        default=GLOBAL_IMPORTANCE_DESCRIPTION,
        init=False,
        repr=False,
    )
    GLOBAL_IMPORTANCE_PERCENT_DESCRIPTION: str = field(
        default=GLOBAL_IMPORTANCE_PERCENT_DESCRIPTION,
        init=False,
        repr=False,
    )

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "label": self.label,
            "global_importance": self.global_importance,
            "global_importance_percent": self.global_importance_percent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalFeatureImportance":
        return cls(
            feature=data.get("feature", ""),
            label=data.get("label", ""),
            global_importance=float(data.get("global_importance", 0.0)),
            global_importance_percent=float(data.get("global_importance_percent", 0.0)),
        )


@dataclass
class InputFeatureValue:
    value: Any = None
    label: str = ""
    display: str = ""

    VALUE_DESCRIPTION: str = field(
        default="Original input feature value",
        init=False,
        repr=False,
    )
    LABEL_DESCRIPTION: str = field(
        default="Human-readable feature label from the shared feature schema",
        init=False,
        repr=False,
    )
    DISPLAY_DESCRIPTION: str = field(
        default="Human-readable feature value representation",
        init=False,
        repr=False,
    )

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "label": self.label,
            "display": self.display,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InputFeatureValue":
        return cls(
            value=data.get("value"),
            label=data.get("label", ""),
            display=data.get("display", ""),
        )


@dataclass
class ExplainResult:
    prediction: float = 0.0
    currency: str = PREDICTION_CURRENCY
    model_name: str = ""
    top_features: List[GlobalFeatureImportance] = field(default_factory=list)
    feature_values: Dict[str, InputFeatureValue] = field(default_factory=dict)

    PREDICTION_DESCRIPTION: str = field(
        default="Predicted car price",
        init=False,
        repr=False,
    )
    CURRENCY_DESCRIPTION: str = field(
        default="Currency code of the predicted price",
        init=False,
        repr=False,
    )
    MODEL_NAME_DESCRIPTION: str = field(
        default="Name of the ML model that produced the prediction",
        init=False,
        repr=False,
    )
    TOP_FEATURES_DESCRIPTION: str = field(
        default=EXPLAIN_TOP_FEATURES_DESCRIPTION,
        init=False,
        repr=False,
    )
    FEATURE_VALUES_DESCRIPTION: str = field(
        default=EXPLAIN_FEATURE_VALUES_DESCRIPTION,
        init=False,
        repr=False,
    )

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "currency": self.currency,
            "model_name": self.model_name,
            "top_features": [f.to_dict() for f in self.top_features],
            "feature_values": {k: v.to_dict() for k, v in self.feature_values.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExplainResult":
        top_features = [
            GlobalFeatureImportance.from_dict(f)
            for f in data.get("top_features", [])
        ]
        feature_values = {
            k: InputFeatureValue.from_dict(v)
            for k, v in data.get("feature_values", {}).items()
        }
        return cls(
            prediction=float(data.get("prediction", 0.0)),
            currency=data.get("currency", PREDICTION_CURRENCY),
            model_name=data.get("model_name", ""),
            top_features=top_features,
            feature_values=feature_values,
        )
