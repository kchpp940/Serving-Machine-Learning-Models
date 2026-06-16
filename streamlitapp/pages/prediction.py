from __future__ import annotations

from typing import Any, Dict

import streamlit as st

import api_client
import state
import ui_helpers
from constants import API_BASE_URL


def _ensure_schema() -> Dict[str, Any]:
    if state.get_schema_cache() is None and state.get_schema_error() is None:
        schema, error = api_client.fetch_schema()
        if error is not None:
            state.set_schema_error(error)
        else:
            state.set_schema_cache(schema)
    return state.get_effective_schema()


def render() -> None:
    st.header("Single Car Price Prediction")

    st.write(
        """
    Enter the car details below and click **Predict Price** to get an estimated price.
    """
    )

    schema = _ensure_schema()
    ui_helpers.ensure_schema_loaded(
        schema,
        state.get_schema_error(),
        state.is_schema_using_fallback(),
        API_BASE_URL,
    )

    st.subheader("Input Car Details")
    car_name = st.text_input("Name of Car", key="pred_car_name")

    inputs = ui_helpers.build_form(schema, key_prefix="pred_")

    if st.button("Predict Price", key="pred_predict_btn"):
        if not car_name:
            state.set_error_message("Please enter the name of the car.")
        else:
            values = {field: inputs[field] for field in schema["feature_order"]}
            prediction, error = api_client.predict(values)
            if error is not None:
                state.set_error_message(error)
            else:
                result = state.PredictionResult(
                    scenario_name=car_name,
                    features=values,
                    prediction=prediction,
                )
                state.add_prediction_result(result)
                state.set_success_message(
                    f"The Price of the {car_name} is {prediction:.2f}$"
                )

    error_msg, success_msg = state.consume_messages()
    if error_msg:
        st.error(error_msg)
    if success_msg:
        st.success(success_msg)
