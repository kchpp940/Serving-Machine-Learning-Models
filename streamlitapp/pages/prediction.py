from __future__ import annotations

import streamlit as st

from car_pricing.api_client import CarPricingApiClient

from ui_helpers import (
    get_default_schema,
    render_schema_form,
    render_schema_status,
    show_error,
    show_success,
    format_currency,
)


@st.cache_data(show_spinner=False)
def _fetch_schema_cached(api_base_url: str, timeout: int):
    client = CarPricingApiClient(base_url=api_base_url, timeout=timeout)
    schema, err = client.fetch_schema()
    if err:
        return None, err
    return schema, None


def render(api_client: CarPricingApiClient) -> None:
    st.header("Car Price Prediction")

    st.write("Enter the details of the car below to get a price prediction.")

    schema, schema_error = _fetch_schema_cached(api_client.base_url, api_client.timeout)
    using_fallback = False
    if schema_error is not None:
        schema = get_default_schema()
        using_fallback = True

    render_schema_status(schema, using_fallback, api_client.base_url)

    st.subheader("Input Car Details")
    car_name = st.text_input("Name of Car", key="pred_car_name")

    inputs = render_schema_form(schema, key_prefix="pred_", use_two_columns=True)

    if st.button("Predict Price", type="primary"):
        if not car_name:
            show_error("Please enter the name of the car.")
            return

        values = {}
        for field in schema["feature_order"]:
            values[field] = inputs[field]

        with st.spinner("Predicting..."):
            prediction, err = api_client.predict(values)

        if err:
            show_error(f"Prediction failed: {err}")
        else:
            show_success(f"The estimated price of the **{car_name}** is **{format_currency(prediction)}**")

            with st.expander("View Input Details"):
                st.write("Features used for prediction:")
                details = {}
                from ui_helpers import field_display_name
                for field in schema["feature_order"]:
                    details[field_display_name(field, schema)] = values[field]
                st.table(details)
