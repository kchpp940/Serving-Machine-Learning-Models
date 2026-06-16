from __future__ import annotations

import streamlit as st

from car_pricing.api_client import API_BASE_URL, ApiError
import state
import ui_helpers


def render() -> None:
    st.header("Single Car Price Prediction")

    st.write(
        """
    Enter the car details below and click **Predict Price** to get an estimated price.
    """
    )

    schema = state.ensure_schema()
    schema_error = state.get_schema_error()

    if schema_error is not None:
        st.warning(
            f"{schema_error} The prediction form is unavailable until the service is reachable."
        )
        return

    if schema is None:
        st.info("Loading schema from the prediction service...")
        return

    st.success(
        f"Loaded schema from {API_BASE_URL} ({len(schema['feature_order'])} features)"
    )

    st.subheader("Input Car Details")
    car_name = st.text_input("Name of Car", key="pred_car_name")

    inputs = ui_helpers.build_form(schema, key_prefix="pred_")

    if st.button("Predict Price", key="pred_predict_btn"):
        if not car_name:
            state.set_error_message("Please enter the name of the car.")
        else:
            values = {field: inputs[field] for field in schema["feature_order"]}
            prediction, error = state.run_predict(values)
            if error is not None:
                state.set_error_message(str(error))
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
