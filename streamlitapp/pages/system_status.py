from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import pandas as pd

from car_pricing.api_client import CarPricingApiClient

from ui_helpers import (
    show_error,
    show_success,
    show_warning,
    show_info,
)


def _format_status_badge(status: str) -> str:
    status_lower = status.lower() if status else "unknown"
    if status_lower == "healthy":
        return "✅ Healthy"
    elif status_lower == "unhealthy":
        return "❌ Unhealthy"
    else:
        return f"ℹ️ {status or 'Unknown'}"


def _display_kv_section(title: str, data: Dict[str, Any]) -> None:
    if not data:
        return
    st.subheader(title)
    df = pd.DataFrame(list(data.items()), columns=["Property", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render(api_client: CarPricingApiClient) -> None:
    st.header("System Status")

    st.write("Monitor the health and status of the prediction service.")

    st.divider()

    st.subheader("Service Health")

    health_col1, health_col2 = st.columns([1, 2])
    with health_col1:
        if st.button("Refresh Status", type="primary"):
            st.rerun()

    with st.spinner("Checking service health..."):
        health_data, health_err = api_client.get_health()
        status_data, status_err = api_client.get_status()
        metadata_data, metadata_err = api_client.get_metadata()

    if health_err:
        show_error(f"Unable to check service health: {health_err}")
        st.metric("API Server", "❌ Unreachable")
    else:
        status_str = health_data.get("status", "unknown")
        st.metric("Service Status", _format_status_badge(status_str))

        model_loaded = health_data.get("model_loaded", False)
        st.metric("Model Loaded", "✅ Yes" if model_loaded else "❌ No")

        if model_loaded:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Name", health_data.get("model_name", "N/A"))
            with col2:
                st.metric("Model Type", health_data.get("model_type", "N/A"))
            with col3:
                st.metric("Features", health_data.get("n_features", "N/A"))

    st.divider()

    status_tab, meta_tab, schema_tab = st.tabs(["Status Details", "Metadata", "Schema Info"])

    with status_tab:
        if status_err:
            show_error(f"Failed to load status: {status_err}")
        elif status_data:
            if isinstance(status_data, dict):
                top_level = {}
                nested = {}
                for key, value in status_data.items():
                    if isinstance(value, dict):
                        nested[key] = value
                    else:
                        top_level[key] = value

                if top_level:
                    _display_kv_section("Overview", top_level)

                for section_title, section_data in nested.items():
                    if isinstance(section_data, dict):
                        _display_kv_section(section_title.replace("_", " ").title(), section_data)
            else:
                st.json(status_data)
        else:
            show_info("No status data available.")

    with meta_tab:
        if metadata_err:
            show_error(f"Failed to load metadata: {metadata_err}")
        elif metadata_data:
            if isinstance(metadata_data, dict):
                top_level = {}
                nested = {}
                for key, value in metadata_data.items():
                    if isinstance(value, dict):
                        nested[key] = value
                    else:
                        top_level[key] = value

                if top_level:
                    _display_kv_section("Model Information", top_level)

                for section_title, section_data in nested.items():
                    if isinstance(section_data, dict):
                        _display_kv_section(section_title.replace("_", " ").title(), section_data)
            else:
                st.json(metadata_data)
        else:
            show_info("No metadata available.")

    with schema_tab:
        from ui_helpers import get_default_schema, field_display_name

        schema_data, schema_err = api_client.fetch_schema()
        if schema_err:
            show_warning(f"Using default schema: {schema_err}")
            schema_data = get_default_schema()

        if schema_data:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Features", len(schema_data.get("feature_order", [])))
                st.metric("Numeric Features", len(schema_data.get("numeric_features", [])))
            with col2:
                st.metric("Categorical Features", len(schema_data.get("categorical_features", [])))
                st.metric("Target Column", schema_data.get("target_column", "N/A"))

            st.subheader("Feature Order")
            features_df = pd.DataFrame(
                [
                    {
                        "Name": f,
                        "Display Name": field_display_name(f, schema_data),
                        "Type": "Numeric" if f in schema_data.get("numeric_features", []) else "Categorical",
                    }
                    for f in schema_data.get("feature_order", [])
                ]
            )
            st.dataframe(features_df, use_container_width=True, hide_index=True)

            st.subheader("Categorical Options")
            cat_options = schema_data.get("categorical_options", {})
            if cat_options:
                for field, options in cat_options.items():
                    with st.expander(field_display_name(field, schema_data)):
                        opts_df = pd.DataFrame(options)
                        st.dataframe(opts_df, use_container_width=True, hide_index=True)
            else:
                show_info("No categorical features.")
        else:
            show_error("No schema data available.")
