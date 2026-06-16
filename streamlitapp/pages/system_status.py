from __future__ import annotations

from typing import Any, Dict, Tuple

import streamlit as st
import pandas as pd

from car_pricing.api_client import (
    CarPricingApiClient,
    HealthStatus,
    ServiceMetadata,
)

from ui_helpers import (
    show_error,
    show_api_error,
    show_success,
    show_warning,
    show_info,
    field_display_name,
)


def _format_status_badge(health: HealthStatus) -> str:
    if health.is_healthy():
        return "✅ Healthy"
    return "❌ Unhealthy"


def _display_kv_section(title: str, data: Dict[str, Any]) -> None:
    if not data:
        return
    st.subheader(title)
    df = pd.DataFrame(list(data.items()), columns=["Property", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _display_metadata_sections(metadata: ServiceMetadata) -> None:
    sections = metadata.flat_sections()
    for title, section_data in sections:
        _display_kv_section(title, section_data)


def _render_health_section(api_client: CarPricingApiClient) -> Tuple[HealthStatus, bool]:
    with st.spinner("Checking service health..."):
        health, health_err = api_client.get_health()

    if health_err:
        show_api_error(health_err)
        st.metric("API Server", "❌ Unreachable")
        return None, False

    if health is None:
        show_error("Failed to get health status.")
        return None, False

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Service Status", _format_status_badge(health))
        st.metric("Model Loaded", "✅ Yes" if health.model_loaded else "❌ No")

    if health.is_healthy():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Name", health.model_name or "N/A")
        with col2:
            st.metric("Model Type", health.model_type or "N/A")
        with col3:
            st.metric("Features", health.n_features or "N/A")

    elif health.error:
        show_warning(f"Model error: {health.error}")

    return health, True


def _render_schema_section(api_client: CarPricingApiClient) -> None:
    schema, schema_err = api_client.fetch_schema()

    if schema_err:
        show_warning(f"Using default schema: {schema_err.display()}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Features", len(schema.get("feature_order", [])))
        st.metric("Numeric Features", len(schema.get("numeric_features", [])))
    with col2:
        st.metric("Categorical Features", len(schema.get("categorical_features", [])))
        st.metric("Target Column", schema.get("target_column", "N/A"))

    st.subheader("Feature Order")
    features_df = pd.DataFrame(
        [
            {
                "Name": f,
                "Display Name": field_display_name(f, schema),
                "Type": "Numeric" if f in schema.get("numeric_features", []) else "Categorical",
            }
            for f in schema.get("feature_order", [])
        ]
    )
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    st.subheader("Categorical Options")
    cat_options = schema.get("categorical_options", {})
    if cat_options:
        for field, options in cat_options.items():
            with st.expander(field_display_name(field, schema)):
                opts_df = pd.DataFrame(options)
                st.dataframe(opts_df, use_container_width=True, hide_index=True)
    else:
        show_info("No categorical features.")


def render(api_client: CarPricingApiClient) -> None:
    st.header("System Status")

    st.write("Monitor the health and status of the prediction service.")

    st.divider()

    st.subheader("Service Health")

    if st.button("Refresh Status", type="primary"):
        st.rerun()

    health, health_ok = _render_health_section(api_client)

    st.divider()

    if not health_ok:
        status_data, status_err = None, health_err
        metadata_data, metadata_err = None, health_err
    else:
        with st.spinner("Loading metadata..."):
            status_data, status_err = api_client.get_status()
            metadata_data, metadata_err = api_client.get_metadata()

    status_tab, meta_tab, schema_tab = st.tabs(["Status Details", "Metadata", "Schema Info"])

    with status_tab:
        if status_err:
            show_api_error(status_err)
        elif status_data:
            _display_metadata_sections(status_data)
        else:
            show_info("No status data available.")

    with meta_tab:
        if metadata_err:
            show_api_error(metadata_err)
        elif metadata_data:
            _display_metadata_sections(metadata_data)
        else:
            show_info("No metadata available.")

    with schema_tab:
        _render_schema_section(api_client)
