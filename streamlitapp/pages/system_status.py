from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import state


def _format_dict_as_table(data: dict, title: str) -> None:
    if not data:
        st.info(f"No {title.lower()} available.")
        return
    df = pd.DataFrame(
        [
            {
                "Key": k,
                "Value": json.dumps(v, indent=2) if isinstance(v, (dict, list)) else v,
            }
            for k, v in data.items()
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render() -> None:
    st.header("System Status")

    st.write(
        f"""
    Check the health, status, and metadata of the prediction service running at
    `{state.API_BASE_URL}`.
    """
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Refresh Status", key="sys_refresh_btn"):
            success, error = state.refresh_service_status()
            if success:
                state.set_success_message("Service status refreshed successfully.")
            else:
                state.set_error_message(f"Failed to refresh service status: {error}")
    with col2:
        refreshed_at = state.get_service_status_refreshed_at()
        if refreshed_at:
            st.caption(f"Last refreshed: {refreshed_at.strftime('%Y-%m-%d %H:%M:%S')}")

    if state.get_service_status() is None:
        state.refresh_service_status()

    error_msg, success_msg = state.consume_messages()
    if error_msg:
        st.error(error_msg)
    if success_msg:
        st.success(success_msg)

    health = state.get_service_health()
    if health is not None:
        st.subheader("Health")
        status_val = health.get("status", "unknown")
        if status_val == "healthy":
            st.success(f"Service is **{status_val}** ✅")
        elif status_val == "unhealthy":
            st.error(f"Service is **{status_val}** ❌")
            if health.get("error"):
                st.write(f"Error: `{health['error']}`")
        else:
            st.warning(f"Service status: **{status_val}**")
        _format_dict_as_table(health, "Health")

    status_data = state.get_service_status()
    if status_data is not None:
        st.subheader("Service Status")
        _format_dict_as_table(status_data, "Status")

    metadata = state.get_service_metadata()
    if metadata is not None:
        st.subheader("Model Metadata")
        _format_dict_as_table(metadata, "Metadata")
