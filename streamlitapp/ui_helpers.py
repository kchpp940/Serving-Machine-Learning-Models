from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

from car_pricing.feature_schema import (
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    FIELD_DISPLAY_NAMES,
    FIELD_DEFAULT_VALUES,
    _display_name,
)


DEFAULT_SCHEMA: Dict[str, Any] = {
    "feature_order": list(FEATURE_ORDER),
    "numeric_features": list(NUMERIC_FEATURES),
    "categorical_features": list(CATEGORICAL_FEATURES),
    "target_column": TARGET_COLUMN,
    "categorical_options": {
        "drivewheel": [
            {"display": "Four Wheel Drive (4WD)", "form_value": "4wd", "model_code": 0},
            {"display": "Front Wheel Drive (FWD)", "form_value": "fwd", "model_code": 1},
            {"display": "Rear Wheel Drive (RWD)", "form_value": "rwd", "model_code": 2},
        ],
        "cylindernumber": [
            {"display": "2 cylinders", "form_value": "two", "model_code": 6},
            {"display": "3 cylinders", "form_value": "three", "model_code": 4},
            {"display": "4 cylinders", "form_value": "four", "model_code": 2},
            {"display": "5 cylinders", "form_value": "five", "model_code": 1},
            {"display": "6 cylinders", "form_value": "six", "model_code": 3},
            {"display": "8 cylinders", "form_value": "eight", "model_code": 0},
            {"display": "12 cylinders", "form_value": "twelve", "model_code": 5},
        ],
    },
    "display_names": dict(FIELD_DISPLAY_NAMES),
    "default_values": dict(FIELD_DEFAULT_VALUES),
}


def get_default_schema() -> Dict[str, Any]:
    return dict(DEFAULT_SCHEMA)


def field_display_name(field_name: str, schema: Optional[Dict[str, Any]] = None) -> str:
    if schema and "display_names" in schema:
        return schema["display_names"].get(field_name, field_name.replace("_", " ").title())
    return FIELD_DISPLAY_NAMES.get(field_name, field_name.replace("_", " ").title())


def field_default_value(field_name: str, schema: Optional[Dict[str, Any]] = None) -> Any:
    if schema and "default_values" in schema:
        if field_name in schema["default_values"]:
            return schema["default_values"][field_name]
    if field_name in FIELD_DEFAULT_VALUES:
        return FIELD_DEFAULT_VALUES[field_name]
    numeric_features = set(schema["numeric_features"]) if schema else set(NUMERIC_FEATURES)
    return 0.0 if field_name in numeric_features else ""


def show_error(message: str) -> None:
    st.error(message)


def show_success(message: str) -> None:
    st.success(message)


def show_warning(message: str) -> None:
    st.warning(message)


def show_info(message: str) -> None:
    st.info(message)


def render_schema_form(
    schema: Dict[str, Any],
    key_prefix: str = "",
    use_two_columns: bool = False,
) -> Dict[str, Any]:
    feature_order = schema["feature_order"]
    numeric_features = set(schema["numeric_features"])
    categorical_features = set(schema["categorical_features"])
    categorical_options = schema.get("categorical_options", {})

    inputs: Dict[str, Any] = {}

    if use_two_columns:
        mid = len(feature_order) // 2
        col1_fields = feature_order[:mid]
        col2_fields = feature_order[mid:]
        col1, col2 = st.columns(2)

        with col1:
            for field in col1_fields:
                inputs[field] = _render_field(
                    field, schema, numeric_features, categorical_features, categorical_options, key_prefix
                )

        with col2:
            for field in col2_fields:
                inputs[field] = _render_field(
                    field, schema, numeric_features, categorical_features, categorical_options, key_prefix
                )
    else:
        for field in feature_order:
            inputs[field] = _render_field(
                field, schema, numeric_features, categorical_features, categorical_options, key_prefix
            )

    return inputs


def _render_field(
    field: str,
    schema: Dict[str, Any],
    numeric_features: set,
    categorical_features: set,
    categorical_options: Dict[str, Any],
    key_prefix: str,
) -> Any:
    label = field_display_name(field, schema)
    key = f"{key_prefix}{field}"

    if field in numeric_features:
        default = field_default_value(field, schema)
        return st.number_input(label, value=float(default), step=0.1, key=key)
    elif field in categorical_features:
        options = categorical_options.get(field, [])
        if not options:
            default = field_default_value(field, schema)
            return st.text_input(label, value=str(default), key=key)
        display_labels = [opt["display"] for opt in options]
        default_value = field_default_value(field, schema)
        default_idx = 0
        for i, opt in enumerate(options):
            if opt["form_value"] == default_value:
                default_idx = i
                break
        selected_idx = st.selectbox(
            label,
            range(len(display_labels)),
            format_func=lambda i, labels=display_labels: labels[i],
            index=default_idx,
            key=key,
        )
        return options[selected_idx]["form_value"]
    else:
        default = field_default_value(field, schema)
        return st.text_input(label, value=str(default), key=key)


def format_currency(value: float, currency: str = "$") -> str:
    return f"{currency}{value:,.2f}"


def format_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def build_result_table(
    scenarios: List[Dict[str, Any]],
    schema: Dict[str, Any],
    sort_by: Optional[str] = None,
    ascending: bool = True,
) -> pd.DataFrame:
    if not scenarios:
        return pd.DataFrame()

    feature_order = schema["feature_order"]
    display_names = {f: field_display_name(f, schema) for f in feature_order}

    rows = []
    for scenario in scenarios:
        row = {"Scenario": scenario.get("name", "Unnamed")}
        for f in feature_order:
            row[display_names[f]] = scenario.get("values", {}).get(f, "")
        row["Predicted Price"] = scenario.get("prediction", "")
        rows.append(row)

    df = pd.DataFrame(rows)

    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    return df


def display_result_table(df: pd.DataFrame) -> None:
    if df.empty:
        show_info("No results to display.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_schema_status(schema: Dict[str, Any], using_fallback: bool, api_base_url: str) -> None:
    if using_fallback:
        show_warning(
            f"Using default schema. The form may not match the server's expectations. "
            f"Could not reach {api_base_url}"
        )
    else:
        show_success(
            f"Loaded schema from {api_base_url} "
            f"({len(schema['feature_order'])} features)"
        )
