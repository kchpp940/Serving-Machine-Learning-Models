from __future__ import annotations

from typing import Dict, List, Any

import streamlit as st
import pandas as pd

from car_pricing.api_client import (
    CarPricingApiClient,
    BatchPredictionResult,
)

from ui_helpers import (
    render_schema_form,
    render_schema_status,
    show_error,
    show_api_error,
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
    return schema, err


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


def _collect_batch_data(scenarios: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
    batch: List[Dict[str, Any]] = []
    names: List[str] = []

    for scenario in scenarios:
        values = {}
        for field in schema["feature_order"]:
            key = f"scen_{scenario['id']}_{field}"
            values[field] = st.session_state.get(key, scenario["values"].get(field))
        scenario["values"] = values
        batch.append(values)
        names.append(scenario["name"])

    return {"batch": batch, "names": names}


def _apply_batch_result(
    scenarios: List[Dict[str, Any]],
    batch_result: BatchPredictionResult,
    batch_info: Dict[str, Any],
) -> None:
    name_to_scenario = {s["name"]: s for s in scenarios}

    for result in batch_result.results:
        if result.scenario_name and result.scenario_name in name_to_scenario:
            scenario = name_to_scenario[result.scenario_name]
            scenario["prediction"] = result.prediction
            scenario["values"] = result.values


def _predict_all(api_client: CarPricingApiClient, schema: Dict[str, Any]) -> None:
    scenarios = st.session_state.scenarios
    if not scenarios:
        show_warning("No scenarios to predict. Add at least one scenario first.")
        return

    batch_info = _collect_batch_data(scenarios, schema)

    with st.spinner(f"Predicting {len(batch_info['batch'])} scenarios..."):
        batch_result, err = api_client.predict_batch(
            batch_info["batch"],
            scenario_names=batch_info["names"],
        )

    if err:
        show_api_error(err)
        return

    if batch_result is None:
        show_error("Failed to get batch prediction results.")
        return

    _apply_batch_result(scenarios, batch_result, batch_info)
    show_success(f"Successfully predicted {len(batch_result)} scenarios.")


def render(api_client: CarPricingApiClient) -> None:
    st.header("Scenario Comparison")

    st.write(
        "Compare multiple car configurations side by side. "
        "Add scenarios, adjust features, and see how predicted prices differ."
    )

    _init_scenarios()

    schema, schema_err = _fetch_schema_cached(api_client.base_url, api_client.timeout)

    render_schema_status(schema, schema_err, api_client.base_url)

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
