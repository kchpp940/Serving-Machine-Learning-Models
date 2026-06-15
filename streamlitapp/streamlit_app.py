import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from car_pricing.api_client import (
    create_client,
    ServiceError,
    DEFAULT_SCHEMA,
    FIELD_DISPLAY_NAMES,
)

SERVICE_TYPE = os.environ.get("API_SERVICE_TYPE", "fastapi")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))


def _get_client():
    return create_client(
        service_type=SERVICE_TYPE,
        base_url=API_BASE_URL,
        timeout=REQUEST_TIMEOUT,
    )


@st.cache_data(show_spinner=False)
def fetch_schema():
    client = _get_client()
    try:
        data = client.get_schema()
        return data, None
    except ServiceError as e:
        return None, e.message
    except Exception as e:
        return None, f"Unexpected error fetching schema: {e}"


def build_form(schema):
    feature_order = schema["feature_order"]
    numeric_features = set(schema["numeric_features"])
    categorical_features = set(schema["categorical_features"])
    categorical_options = schema["categorical_options"]

    inputs = {}
    for field in feature_order:
        label = FIELD_DISPLAY_NAMES.get(field, field.replace("_", " ").title())
        if field in numeric_features:
            inputs[field] = st.number_input(label, value=0.0, step=0.1)
        elif field in categorical_features:
            options = categorical_options.get(field, [])
            if not options:
                inputs[field] = st.text_input(label)
            else:
                display_labels = [opt["display"] for opt in options]
                selected_idx = st.selectbox(label, range(len(display_labels)), format_func=lambda i: display_labels[i])
                inputs[field] = options[selected_idx]["form_value"]
        else:
            inputs[field] = st.text_input(label)
    return inputs


def main():
    st.title("Car Price Prediction Web App")

    st.write("""
    ## About

    **This Streamlit App utilizes a Machine Learning model served as an API to predict the price of a car based on certain features.**

    """)

    schema, schema_error = fetch_schema()
    using_fallback = False
    if schema_error is not None:
        st.warning(f"{schema_error} Using default schema. The form may not match the server's expectations.")
        schema = DEFAULT_SCHEMA
        using_fallback = True
    else:
        st.success(f"Loaded schema from {API_BASE_URL} ({len(schema['feature_order'])} features)")

    st.header("Input Car Details")
    names = st.text_input("Name of Car")

    inputs = build_form(schema)

    if st.button("Predict Price"):
        if not names:
            st.error("Please enter the name of the car.")
            return

        values = {}
        for field in schema["feature_order"]:
            values[field] = inputs[field]

        client = _get_client()
        try:
            body = client.predict(values)
            prediction = body.get("prediction")
            if prediction is None:
                st.error(f"Unexpected response format from server: {body}")
            else:
                st.success(f"The Price of the {names} is {prediction:.2f}$")
        except ServiceError as e:
            st.error(e.message)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
