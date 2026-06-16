from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from car_pricing.api_client import CarPricingApiClient

from pages import render_prediction, render_scenario_compare, render_system_status


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))

PAGE_MAP = {
    "Prediction": render_prediction,
    "Scenario Comparison": render_scenario_compare,
    "System Status": render_system_status,
}


def _init_api_client() -> CarPricingApiClient:
    if "api_client" not in st.session_state:
        st.session_state.api_client = CarPricingApiClient(
            base_url=API_BASE_URL,
            timeout=REQUEST_TIMEOUT,
        )
    return st.session_state.api_client


def _configure_page() -> None:
    st.set_page_config(
        page_title="Car Price Prediction",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _render_sidebar() -> str:
    with st.sidebar:
        st.title("🚗 Car Pricing")
        st.caption("Machine Learning Prediction Service")

        st.divider()

        page = st.radio(
            "Navigation",
            list(PAGE_MAP.keys()),
            label_visibility="collapsed",
        )

        st.divider()

        with st.expander("API Configuration", expanded=False):
            api_url = st.text_input(
                "API Base URL",
                value=API_BASE_URL,
                key="sidebar_api_url",
            )
            api_timeout = st.number_input(
                "Request Timeout (seconds)",
                value=REQUEST_TIMEOUT,
                min_value=1,
                max_value=120,
                step=1,
                key="sidebar_timeout",
            )

            if st.button("Apply Configuration", type="secondary"):
                st.session_state.api_client = CarPricingApiClient(
                    base_url=api_url,
                    timeout=int(api_timeout),
                )
                st.success("Configuration updated!")
                st.rerun()

        st.divider()

        st.caption(f"API: {st.session_state.api_client.base_url}")

    return page


def _render_footer() -> None:
    st.divider()
    st.caption(
        "Car Price Prediction Web App — Powered by Machine Learning"
    )


def main() -> None:
    _configure_page()
    api_client = _init_api_client()
    page = _render_sidebar()

    st.title("Car Price Prediction Web App")

    st.markdown(
        """
    **This Streamlit App utilizes a Machine Learning model served as an API 
    to predict the price of a car based on certain features.**
    """
    )

    render_func = PAGE_MAP.get(page)
    if render_func:
        render_func(api_client)

    _render_footer()


if __name__ == "__main__":
    main()
