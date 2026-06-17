import os
import streamlit as st
import requests as re

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 10

DEFAULT_SCHEMA = {
    "feature_order": ["year", "km_driven", "fuel", "seller_type", "transmission", "owner", "mileage", "engine", "max_power", "seats", "brand", "individual"],
    "numeric_features": ["year", "km_driven", "mileage", "engine", "max_power", "seats", "individual"],
    "categorical_features": ["fuel", "seller_type", "transmission", "owner", "brand"],
    "target_column": "selling_price",
    "categorical_options": {
        "fuel": [
            {"display": "Diesel", "form_value": "Diesel"},
            {"display": "Petrol", "form_value": "Petrol"},
            {"display": "CNG", "form_value": "CNG"},
            {"display": "LPG", "form_value": "LPG"},
            {"display": "Electric", "form_value": "Electric"},
        ],
        "seller_type": [
            {"display": "Individual", "form_value": "Individual"},
            {"display": "Dealer", "form_value": "Dealer"},
            {"display": "Trustmark Dealer", "form_value": "Trustmark Dealer"},
        ],
        "transmission": [
            {"display": "Manual", "form_value": "Manual"},
            {"display": "Automatic", "form_value": "Automatic"},
        ],
        "owner": [
            {"display": "First Owner", "form_value": "First Owner"},
            {"display": "Second Owner", "form_value": "Second Owner"},
            {"display": "Third Owner", "form_value": "Third Owner"},
            {"display": "Fourth & Above Owner", "form_value": "Fourth & Above Owner"},
            {"display": "Test Drive Car", "form_value": "Test Drive Car"},
        ],
        "brand": [],
    },
}

FIELD_DISPLAY_NAMES = {
    "year": "Year of Manufacture",
    "km_driven": "Kilometers Driven",
    "fuel": "Fuel Type",
    "seller_type": "Seller Type",
    "transmission": "Transmission",
    "owner": "Number of Owners",
    "mileage": "Mileage (km/l)",
    "engine": "Engine (CC)",
    "max_power": "Max Power (bhp)",
    "seats": "Number of Seats",
    "brand": "Brand Name",
    "individual": "Individual Seller (0 = Dealer, 1 = Individual)",
}


@st.cache_data(show_spinner=False)
def fetch_schema():
    url = f"{API_BASE_URL.rstrip('/')}/schema"
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        required = ["feature_order", "numeric_features", "categorical_features", "categorical_options"]
        missing = [k for k in required if k not in data]
        if missing:
            return None, f"Schema missing fields: {', '.join(missing)}"
        return data, None
    except re.exceptions.ConnectionError:
        return None, "Unable to connect to the prediction service to fetch schema."
    except re.exceptions.Timeout:
        return None, "Schema request timed out."
    except re.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            pass
        return None, f"Server returned error when fetching schema: {detail or str(e)}"
    except ValueError:
        return None, "Schema response was not valid JSON."
    except Exception as e:
        return None, f"Unexpected error fetching schema: {str(e)}"


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

        try:
            url = f"{API_BASE_URL.rstrip('/')}/predict"
            res = re.post(url, json=values, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            body = res.json()
            prediction = body.get("prediction")
            if prediction is None:
                st.error(f"Unexpected response format from server: {body}")
                return
            st.success(f"The Price of the {names} is {prediction:.2f}$")
        except re.exceptions.ConnectionError:
            st.error("Unable to connect to the prediction service. Please check that the API server is running.")
        except re.exceptions.Timeout:
            st.error("The request to the prediction service timed out. Please try again later.")
        except re.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = res.json().get("detail", "")
            except Exception:
                pass
            st.error(f"Server returned an error ({res.status_code}): {detail or str(e)}")
        except ValueError:
            st.error("The server returned an invalid response. Please try again later.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
