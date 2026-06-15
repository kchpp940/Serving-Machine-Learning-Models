import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from car_pricing.api_client import (
    create_client,
    ServiceError,
    ErrorCategory,
    SchemaInfo,
    DEFAULT_FALLBACK_SCHEMA,
)


SERVICE_TYPE = os.environ.get("API_SERVICE_TYPE", "fastapi")
BASE_URL = os.environ.get("API_BASE_URL")
TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))


FIELD_DISPLAY_NAMES = {
    "enginesize": "Engine Size",
    "curbweight": "Curb Weight",
    "horsepower": "Horsepower",
    "highwaympg": "Highway Miles Per Gallon",
    "carwidth": "Car Width",
    "wheelbase": "Wheel Base",
    "drivewheel": "Drive Wheel",
    "citympg": "City Miles Per Gallon",
    "boreratio": "Bore Ratio",
    "cylindernumber": "Number of Cylinders",
}


def _get_client():
    return create_client(
        service_type=SERVICE_TYPE,
        base_url=BASE_URL,
        timeout=TIMEOUT,
    )


@st.cache_data(show_spinner=False)
def fetch_schema():
    client = _get_client()
    try:
        schema = client.get_schema()
        return schema, None
    except ServiceError as e:
        return None, e.message
    except Exception as e:
        return None, f"获取 Schema 时发生未知错误: {str(e)}"


def build_form(schema: SchemaInfo):
    feature_order = schema.feature_order
    numeric_features = set(schema.numeric_features)
    categorical_features = set(schema.categorical_features)
    categorical_options = schema.categorical_options

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
                selected_idx = st.selectbox(
                    label,
                    range(len(display_labels)),
                    format_func=lambda i, dl=display_labels: dl[i],
                )
                inputs[field] = options[selected_idx]["form_value"]
        else:
            inputs[field] = st.text_input(label)
    return inputs


def do_predict(features):
    client = _get_client()
    try:
        result = client.predict(features)
        return result.prediction, None
    except ServiceError as e:
        return None, e.message
    except Exception as e:
        return None, f"预测时发生未知错误: {str(e)}"


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
        schema = DEFAULT_FALLBACK_SCHEMA
        using_fallback = True
    else:
        base_url = BASE_URL or f"http://localhost:8000"
        st.success(f"Loaded schema from {base_url} ({len(schema.feature_order)} features)")

    st.header("Input Car Details")
    names = st.text_input("Name of Car")

    inputs = build_form(schema)

    if st.button("Predict Price"):
        if not names:
            st.error("Please enter the name of the car.")
            return

        values = {field: inputs[field] for field in schema.feature_order}
        prediction, error = do_predict(values)

        if error is not None:
            st.error(error)
        else:
            st.success(f"The Price of the {names} is {prediction:.2f}$")


if __name__ == "__main__":
    main()
