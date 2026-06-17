import streamlit as st

from car_pricing.api_client import CarPricingClient, ApiError
from car_pricing.feature_schema import FIELD_DISPLAY_NAMES

client = CarPricingClient()


@st.cache_data(show_spinner=False)
def fetch_schema():
    return client.fetch_schema()


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
    if isinstance(schema_error, ApiError):
        st.warning(f"{schema_error.message} Using default schema. The form may not match the server's expectations.")
        schema = CarPricingClient.build_fallback_schema()
    else:
        st.success(f"Loaded schema from {client.base_url} ({len(schema['feature_order'])} features)")

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

        prediction, err = client.predict(values)
        if isinstance(err, ApiError):
            if err.source == "connection":
                st.error(f"{err.message} {err.detail or ''}")
            elif err.source == "http" and err.status_code is not None:
                st.error(f"{err.message} (status {err.status_code}): {err.detail or ''}")
            else:
                st.error(err.display())
        else:
            st.success(f"The Price of the {names} is {prediction:.2f}$")


if __name__ == "__main__":
    main()
