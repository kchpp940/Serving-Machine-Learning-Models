from __future__ import annotations

import pandas as pd
import streamlit as st

from car_pricing.api_client import API_BASE_URL, fetch_schema, predict
import state
import ui_helpers


def _ensure_schema():
    if state.has_schema():
        return state.get_schema_cache(), None
    if state.get_schema_error() is not None:
        return None, state.get_schema_error()
    data, error = fetch_schema()
    if error is not None:
        state.set_schema_error(error)
        return None, error
    state.set_schema_cache(data)
    return data, None


def _build_results_dataframe():
    results = state.get_prediction_results()
    if not results:
        return None
    schema = state.get_schema_cache()
    if schema is None:
        return None
    rows = []
    for r in results:
        row = {"Scenario": r.scenario_name, "Predicted Price ($)": f"{r.prediction:.2f}"}
        for f in schema["feature_order"]:
            row[ui_helpers.field_label(f)] = r.features.get(f, "")
        row["Created At"] = r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        rows.append(row)
    return pd.DataFrame(rows)


def render() -> None:
    st.header("Scenario Comparison")

    st.write(
        """
    Build multiple car configuration scenarios, predict prices for each,
    and compare the results side by side in a table.
    """
    )

    schema, schema_error = _ensure_schema()

    if schema_error is not None:
        st.warning(
            f"{schema_error} Scenario comparison is unavailable until the service is reachable."
        )
        return

    if schema is None:
        st.info("Loading schema from the prediction service...")
        return

    st.success(
        f"Loaded schema from {API_BASE_URL} ({len(schema['feature_order'])} features)"
    )

    st.subheader("Add New Scenario")
    scenario_name = st.text_input("Scenario Name", key="cmp_scenario_name")
    inputs = ui_helpers.build_form(schema, key_prefix="cmp_")

    col_add, col_clear = st.columns(2)
    with col_add:
        if st.button("Predict & Add to Comparison", key="cmp_add_btn"):
            if not scenario_name:
                state.set_error_message("Please enter a scenario name.")
            else:
                values = {field: inputs[field] for field in schema["feature_order"]}
                prediction, error = predict(values)
                if error is not None:
                    state.set_error_message(str(error))
                else:
                    result = state.PredictionResult(
                        scenario_name=scenario_name,
                        features=values,
                        prediction=prediction,
                    )
                    state.add_prediction_result(result)
                    state.add_scenario(scenario_name)
                    state.set_success_message(
                        f"Added scenario '{scenario_name}' with predicted price {prediction:.2f}$"
                    )
    with col_clear:
        if st.button("Clear All Results", key="cmp_clear_btn"):
            state.clear_prediction_results()
            state.clear_scenarios()
            state.set_success_message("All scenarios cleared.")

    error_msg, success_msg = state.consume_messages()
    if error_msg:
        st.error(error_msg)
    if success_msg:
        st.success(success_msg)

    st.subheader("Comparison Results")
    df = _build_results_dataframe()
    if df is None or df.empty:
        st.info("No prediction results yet. Add a scenario above to see results here.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

        scenarios = state.get_scenario_list()
        if scenarios:
            st.subheader("Remove Scenario")
            to_remove = st.selectbox(
                "Select a scenario to remove",
                options=[""] + scenarios,
                key="cmp_remove_select",
            )
            if st.button("Remove Selected", key="cmp_remove_btn"):
                if to_remove:
                    results = state.get_prediction_results()
                    for i, r in enumerate(results):
                        if r.scenario_name == to_remove:
                            state.remove_prediction_result(i)
                            break
                    state.remove_scenario(to_remove)
                    st.rerun()
