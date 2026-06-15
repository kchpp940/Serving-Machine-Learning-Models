import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests as re

from car_pricing.versioning import build_default_schema_dict

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))


def _fallback_schema_dict() -> dict:
    return build_default_schema_dict(include_encoders=False)


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


def _field_display_name(schema: dict, field_name: str) -> str:
    display_names = schema.get("display_names", {})
    if field_name in display_names:
        return display_names[field_name]
    return field_name.replace("_", " ").title()


def _field_default_value(schema: dict, field_name: str):
    default_values = schema.get("default_values", {})
    if field_name in default_values:
        return default_values[field_name]
    numeric_set = set(schema.get("numeric_features", []))
    return 0.0 if field_name in numeric_set else ""


def build_form(schema):
    feature_order = schema["feature_order"]
    numeric_features = set(schema["numeric_features"])
    categorical_features = set(schema["categorical_features"])
    categorical_options = schema.get("categorical_options", {})

    inputs = {}
    for field in feature_order:
        label = _field_display_name(schema, field)
        default_val = _field_default_value(schema, field)
        if field in numeric_features:
            inputs[field] = st.number_input(label, value=float(default_val), step=0.1)
        elif field in categorical_features:
            options = categorical_options.get(field, [])
            if not options:
                inputs[field] = st.text_input(label, value=str(default_val))
            else:
                display_labels = [opt["display"] for opt in options]
                default_index = 0
                for i, opt in enumerate(options):
                    if opt["form_value"] == default_val:
                        default_index = i
                        break
                selected_idx = st.selectbox(
                    label,
                    range(len(display_labels)),
                    index=default_index,
                    format_func=lambda i: display_labels[i],
                )
                inputs[field] = options[selected_idx]["form_value"]
        else:
            inputs[field] = st.text_input(label, value=str(default_val))
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
        schema = _fallback_schema_dict()
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

        url = f"{API_BASE_URL.rstrip('/')}/predict"
        try:
            res = re.post(url, json=values, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            body = res.json()
            prediction = body.get("prediction")
            if prediction is None:
                st.error(f"Unexpected response format from server: {body}")
            else:
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
