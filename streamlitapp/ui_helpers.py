from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def field_label(field_name: str) -> str:
    return field_name.replace("_", " ").title()


def build_form(schema: Dict[str, Any], key_prefix: str = "") -> Dict[str, Any]:
    feature_order = schema["feature_order"]
    numeric_features = set(schema["numeric_features"])
    categorical_features = set(schema["categorical_features"])
    categorical_options = schema.get("categorical_options", {})

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
                display_labels = [opt.get("display", opt.get("form_value", str(opt))) for opt in options]
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
