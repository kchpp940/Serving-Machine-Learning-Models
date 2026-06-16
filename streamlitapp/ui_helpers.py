from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from constants import FIELD_DISPLAY_NAMES


def field_label(field: str) -> str:
    return FIELD_DISPLAY_NAMES.get(field, field.replace("_", " ").title())


def build_form(schema: Dict[str, Any], key_prefix: str = "") -> Dict[str, Any]:
    feature_order = schema["feature_order"]
    numeric_features = set(schema["numeric_features"])
    categorical_features = set(schema["categorical_features"])
    categorical_options = schema["categorical_options"]

    inputs: Dict[str, Any] = {}
    for field in feature_order:
        label = field_label(field)
        widget_key = f"{key_prefix}{field}" if key_prefix else field
        if field in numeric_features:
            inputs[field] = st.number_input(label, value=0.0, step=0.1, key=widget_key)
        elif field in categorical_features:
            options = categorical_options.get(field, [])
            if not options:
                inputs[field] = st.text_input(label, key=widget_key)
            else:
                display_labels = [opt["display"] for opt in options]
                selected_idx = st.selectbox(
                    label,
                    range(len(display_labels)),
                    format_func=lambda i: display_labels[i],
                    key=widget_key,
                )
                inputs[field] = options[selected_idx]["form_value"]
        else:
            inputs[field] = st.text_input(label, key=widget_key)
    return inputs


def ensure_schema_loaded(schema: Dict[str, Any], schema_error: str | None, using_fallback: bool, api_base_url: str) -> None:
    if schema_error is not None:
        st.warning(
            f"{schema_error} Using default schema. The form may not match the server's expectations."
        )
    else:
        st.success(
            f"Loaded schema from {api_base_url} ({len(schema['feature_order'])} features)"
        )
