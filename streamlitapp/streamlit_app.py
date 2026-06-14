import os
import sys
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from car_pricing.api_client import (
    PredictionAPIClient,
    PredictionResult,
    HealthResult,
    SchemaResult,
    APIHTTPError,
    APIConnectionError,
    APITimeoutError,
    APIInvalidResponseError,
)

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

DEFAULT_SCHEMA = {
    "feature_order": [
        "enginesize", "curbweight", "horsepower", "highwaympg",
        "carwidth", "wheelbase", "drivewheel", "citympg",
        "boreratio", "cylindernumber",
    ],
    "numeric_features": [
        "enginesize", "curbweight", "horsepower", "highwaympg",
        "carwidth", "wheelbase", "citympg", "boreratio",
    ],
    "categorical_features": ["drivewheel", "cylindernumber"],
    "target_column": "price",
    "categorical_options": {
        "drivewheel": [
            {"display": "Four Wheel Drive (4WD)", "form_value": "4wd", "model_code": 0},
            {"display": "Front Wheel Drive (FWD)", "form_value": "fwd", "model_code": 1},
            {"display": "Rear Wheel Drive (RWD)", "form_value": "rwd", "model_code": 2},
        ],
        "cylindernumber": [
            {"display": "2 cylinders", "form_value": "two", "model_code": 6},
            {"display": "3 cylinders", "form_value": "three", "model_code": 4},
            {"display": "4 cylinders", "form_value": "four", "model_code": 2},
            {"display": "5 cylinders", "form_value": "five", "model_code": 1},
            {"display": "6 cylinders", "form_value": "six", "model_code": 3},
            {"display": "8 cylinders", "form_value": "eight", "model_code": 0},
            {"display": "12 cylinders", "form_value": "twelve", "model_code": 5},
        ],
    },
}


def get_api_client(base_url: str = None, timeout: int = None) -> PredictionAPIClient:
    return PredictionAPIClient(base_url=base_url, timeout=timeout)


def get_api_config():
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    if "request_timeout" not in st.session_state:
        st.session_state.request_timeout = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))
    return st.session_state.api_base_url, st.session_state.request_timeout


def check_api_health(api_base_url, timeout):
    client = get_api_client(base_url=api_base_url, timeout=timeout)
    try:
        result = client.health_check()
        return result.status == "ok", result.raw_response
    except Exception:
        return False, None


@st.cache_data(show_spinner=False)
def fetch_schema(api_base_url, timeout):
    client = get_api_client(base_url=api_base_url, timeout=timeout)
    try:
        result = client.get_schema()
        return result.raw_response, None
    except Exception as e:
        return None, PredictionAPIClient.format_error(e)


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

    api_base_url, request_timeout = get_api_config()

    with st.sidebar:
        st.header("⚙️ API Configuration")
        new_api_url = st.text_input(
            "API Base URL",
            value=api_base_url,
            help="The base URL of the prediction API server",
            placeholder="e.g., http://localhost:8000 or https://your-api.herokuapp.com"
        )
        new_timeout = st.number_input(
            "Request Timeout (seconds)",
            min_value=1,
            max_value=120,
            value=request_timeout,
            help="Maximum time to wait for API response"
        )

        if new_api_url != api_base_url or new_timeout != request_timeout:
            st.session_state.api_base_url = new_api_url.rstrip('/')
            st.session_state.request_timeout = int(new_timeout)
            fetch_schema.clear()
            st.experimental_rerun()

        st.divider()

        st.subheader("🔧 API Connection")
        if st.button("Check Connection", use_container_width=True):
            with st.spinner("Checking API connection..."):
                healthy, health_data = check_api_health(st.session_state.api_base_url, st.session_state.request_timeout)
                if healthy:
                    st.success("✅ API is reachable")
                    if health_data:
                        st.info(f"Service: {health_data.get('service', 'N/A')}\n\nVersion: {health_data.get('version', 'N/A')}\n\nModel Loaded: {'Yes' if health_data.get('model_loaded') else 'No'}")
                else:
                    st.error("❌ API is not reachable")
                    st.caption("Please check the API URL and ensure the server is running.")

        st.divider()
        st.caption(f"Current API: {st.session_state.api_base_url}")

    st.write("""
    ## About

    **This Streamlit App utilizes a Machine Learning model served as an API to predict the price of a car based on certain features.**

    """)

    schema, schema_error = fetch_schema(st.session_state.api_base_url, st.session_state.request_timeout)
    using_fallback = False
    if schema_error is not None:
        st.warning(f"{schema_error} Using default schema. The form may not match the server's expectations.")
        schema = DEFAULT_SCHEMA
        using_fallback = True
    else:
        st.success(f"Loaded schema from API ({len(schema['feature_order'])} features)")

    st.header("Input Car Details")
    names = st.text_input("Name of Car")

    inputs = build_form(schema)

    if st.button("Predict Price", type="primary"):
        if not names:
            st.error("Please enter the name of the car.")
            return

        values = {}
        for field in schema["feature_order"]:
            values[field] = inputs[field]

        client = get_api_client(
            base_url=st.session_state.api_base_url,
            timeout=st.session_state.request_timeout,
        )

        with st.spinner("Predicting..."):
            try:
                result = client.predict(values)
                st.success(f"The Price of the {names} is **{result.prediction:.2f} {result.currency}**")
            except Exception as e:
                st.error(PredictionAPIClient.format_error(e))


if __name__ == "__main__":
    main()
