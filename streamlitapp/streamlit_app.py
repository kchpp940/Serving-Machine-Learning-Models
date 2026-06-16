from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

import state
from pages import prediction as page_prediction
from pages import scenario_compare as page_scenario_compare
from pages import system_status as page_system_status


_PAGE_RENDERERS = {
    state.PAGE_PREDICTION: page_prediction.render,
    state.PAGE_SCENARIO_COMPARE: page_scenario_compare.render,
    state.PAGE_SYSTEM_STATUS: page_system_status.render,
}


def _render_sidebar_nav() -> None:
    with st.sidebar:
        st.title("Navigation")
        selected = st.radio(
            "Go to",
            state.ALL_PAGES,
            index=state.ALL_PAGES.index(state.get_current_page()),
            key="nav_radio",
        )
        if selected != state.get_current_page():
            state.set_current_page(selected)
            st.rerun()

        st.divider()
        if st.button("Reset All Session State", key="sidebar_reset_btn"):
            state.reset_all_state()
            st.rerun()


def main() -> None:
    st.set_page_config(page_title=state.APP_TITLE, layout="wide")
    state.init_state()

    st.title(state.APP_TITLE)
    st.markdown(
        """
    **This Streamlit App utilizes a Machine Learning model served as an API
    to predict the price of a car based on certain features.**
    """
    )

    _render_sidebar_nav()

    current_page = state.get_current_page()
    renderer = _PAGE_RENDERERS.get(current_page)
    if renderer is not None:
        renderer()
    else:
        st.error(f"Unknown page: {current_page}")


if __name__ == "__main__":
    main()
