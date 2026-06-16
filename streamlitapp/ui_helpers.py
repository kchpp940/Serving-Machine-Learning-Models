from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

from car_pricing.api_client import ApiError


def field_display_name(field_name: str, schema: Dict[str, Any]) -> str:
    if "display_names" in schema:
        return schema["display_names"].get(field_name, field_name.replace("_", " ").title())
    return field_name.replace("_", " ").title()


def field_default_value(field_name: str, schema: Dict[str, Any]) -> Any:
    if "default_values" in schema and field_name in schema["default_values"]:
        return schema["default_values"][field_name]
    numeric_features = set(schema.get("numeric_features", []))
    return 0.0 if field_name in numeric_features else ""


def show_error(message: str) -> None:
    st.error(message)


def show_api_error(err: ApiError) -> None:
    st.error(err.display())


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


def render_schema_status(schema: Dict[str, Any], fetch_err: Optional[ApiError], api_base_url: str) -> None:
    if fetch_err is not None:
        show_warning(
            f"{fetch_err.display()} Using default schema. The form may not match the server's expectations. "
            f"Could not reach {api_base_url}"
        )
    else:
        show_success(
            f"Loaded schema from {api_base_url} "
            f"({len(schema['feature_order'])} features)"
        )
