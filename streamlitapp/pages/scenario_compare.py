from __future__ import annotations

from typing import Dict, List, Any

import streamlit as st
import pandas as pd

from car_pricing.api_client import CarPricingApiClient

from ui_helpers import (
    get_default_schema,
    render_schema_form,
    render_schema_status,
    show_error,
    show_success,
    show_info,
    show_warning,
    build_result_table,
    display_result_table,
    format_currency,
    field_display_name,
)


@st.cache_data(show_spinner=False)
def _fetch_schema_cached(api_base_url: str, timeout: int):
    client = CarPricingApiClient(base_url=api_base_url, timeout=timeout)
    schema, err = client.fetch_schema()
    if err:
        return None, err
    return schema, None


def _init_scenarios() -> None:
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []
    if "scenario_counter" not in st.session_state:
        st.session_state.scenario_counter = 0


def _add_scenario() -> None:
    st.session_state.scenario_counter += 1
    scenario_id = st.session_state.scenario_counter
    st.session_state.scenarios.append(
        {
            "id": scenario_id,
            "name": f"Scenario {scenario_id}",
            "values": {},
            "prediction": None,
        }
    )


def _remove_scenario(scenario_id: int) -> None:
    st.session_state.scenarios = [
        s for s in st.session_state.scenarios if s["id"] != scenario_id
    ]


def _update_scenario_name(scenario_id: int, name: str) -> None:
    for s in st.session_state.scenarios:
        if s["id"] == scenario_id:
            s["name"] = name
            break


def _predict_all(api_client: CarPricingApiClient, schema: Dict[str, Any]) -> None:
    scenarios = st.session_state.scenarios
    if not scenarios:
        show_warning("No scenarios to predict. Add at least one scenario first.")
        return

    success_count = 0
    for i, scenario in enumerate(scenarios):
        values = {}
        for field in schema["feature_order"]:
            key = f"scen_{scenario['id']}_{field}"
            values[field] = st.session_state.get(key, scenario["values"].get(field))

        scenario["values"] = values
        prediction, err = api_client.predict(values)
        if err:
            scenario["prediction"] = None
            scenario["error"] = err
        else:
            scenario["prediction"] = prediction
            scenario["error"] = None
            success_count += 1

    if success_count == len(scenarios):
        show_success(f"Successfully predicted {success_count} scenarios.")
    elif success_count > 0:
        show_warning(f"Predicted {success_count} of {len(scenarios)} scenarios. Some had errors.")
    else:
        show_error("All predictions failed.")


def render(api_client: CarPricingApiClient) -> None:
    st.header("Scenario Comparison")

    st.write(
        "Compare multiple car configurations side by side. "
        "Add scenarios, adjust features, and see how predicted prices differ."
    )

    _init_scenarios()

    schema, schema_error = _fetch_schema_cached(api_client.base_url, api_client.timeout)
    using_fallback = False
    if schema_error is not None:
        schema = get_default_schema()
        using_fallback = True

    render_schema_status(schema, using_fallback, api_client.base_url)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.button("Add Scenario", on_click=_add_scenario, type="primary")
    with col2:
        st.button("Predict All", on_click=_predict_all, args=(api_client, schema), type="secondary")

    scenarios = st.session_state.scenarios

    if not scenarios:
        show_info("No scenarios yet. Click 'Add Scenario' to start comparing.")
        return

    st.subheader("Scenarios")

    all_predictions = [s["prediction"] for s in scenarios if s["prediction"] is not None]
    if all_predictions:
        min_price = min(all_predictions)
        max_price = max(all_predictions)
        price_diff = max_price - min_price

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Lowest Price", format_currency(min_price))
        with metric_col2:
            st.metric("Highest Price", format_currency(max_price))
        with metric_col3:
            st.metric("Price Difference", format_currency(price_diff))

    for scenario in scenarios:
        scenario_id = scenario["id"]
        with st.expander(f"{scenario['name']}", expanded=True):
            name_col, del_col = st.columns([4, 1])
            with name_col:
                new_name = st.text_input(
                    "Scenario Name",
                    value=scenario["name"],
                    key=f"scen_name_{scenario_id}",
                    label_visibility="collapsed",
                )
                if new_name != scenario["name"]:
                    _update_scenario_name(scenario_id, new_name)
            with del_col:
                st.button(
                    "Remove",
                    key=f"scen_del_{scenario_id}",
                    on_click=_remove_scenario,
                    args=(scenario_id,),
                )

            render_schema_form(schema, key_prefix=f"scen_{scenario_id}_", use_two_columns=True)

            if scenario.get("prediction") is not None:
                st.success(f"Predicted Price: {format_currency(scenario['prediction'])}")
            elif scenario.get("error"):
                st.error(f"Prediction failed: {scenario['error']}")

    if all_predictions:
        st.subheader("Comparison Results")

        sort_col1, sort_col2 = st.columns([2, 1])
        with sort_col1:
            sort_options = ["Scenario"] + [field_display_name(f, schema) for f in schema["feature_order"]] + ["Predicted Price"]
            sort_by = st.selectbox("Sort by", sort_options, index=len(sort_options) - 1)
        with sort_col2:
            sort_ascending = st.radio("Order", ["Ascending", "Descending"], horizontal=True) == "Ascending"

        result_df = build_result_table(
            scenarios,
            schema,
            sort_by=sort_by,
            ascending=sort_ascending,
        )
        display_result_table(result_df)
