import os
import streamlit as st
import requests as re
from datetime import datetime

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
BENTOML_BASE_URL = os.environ.get("BENTOML_BASE_URL", "http://localhost:3000")
REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))

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


def _get_json(url, label):
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            pass
        return None, f"{label} returned HTTP {resp.status_code}: {detail or resp.text}"
    except re.exceptions.ConnectionError:
        return None, f"Cannot connect to {label} at {url}"
    except re.exceptions.Timeout:
        return None, f"{label} request timed out"
    except ValueError:
        return None, f"{label} returned invalid JSON"
    except Exception as e:
        return None, f"{label} error: {str(e)}"


@st.cache_data(show_spinner=False)
def fetch_fastapi_status():
    return _get_json(f"{API_BASE_URL.rstrip('/')}/status", "FastAPI /status")


@st.cache_data(show_spinner=False)
def fetch_fastapi_health():
    return _get_json(f"{API_BASE_URL.rstrip('/')}/health", "FastAPI /health")


@st.cache_data(show_spinner=False)
def fetch_bentoml_status():
    return _get_json(f"{BENTOML_BASE_URL.rstrip('/')}/status", "BentoML /status")


@st.cache_data(show_spinner=False)
def fetch_bentoml_health():
    return _get_json(f"{BENTOML_BASE_URL.rstrip('/')}/health", "BentoML /health")


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


def render_predict_page():
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


def _render_health_badge(name, health_data, health_error):
    if health_error:
        st.error(f"**{name}**: ❌ Unreachable — {health_error}")
        return False
    status = health_data.get("status", "unknown") if health_data else "unknown"
    model_loaded = health_data.get("model_loaded", False) if health_data else False
    if status == "healthy" and model_loaded:
        st.success(f"**{name}**: ✅ Healthy (model loaded)")
        return True
    elif status == "healthy":
        st.warning(f"**{name}**: ⚠️ Healthy but model not loaded")
        return False
    else:
        err = health_data.get("error", "unknown error") if health_data else "unknown"
        st.error(f"**{name}**: ❌ Unhealthy — {err}")
        return False


def _render_status_card(name, status_data, status_error):
    if status_error:
        st.error(f"**{name} Status** — unavailable: {status_error}")
        return None
    with st.expander(f"{name} Status Details", expanded=True):
        st.json(status_data)
    return status_data


def _compare_schemas(fastapi_status, bentoml_status):
    st.subheader("Schema Consistency Check")
    fa_sv = fastapi_status.get("schema_version", "N/A") if fastapi_status else "N/A"
    bm_sv = bentoml_status.get("schema_version", "N/A") if bentoml_status else "N/A"
    fa_dv = fastapi_status.get("data_version", "N/A") if fastapi_status else "N/A"
    bm_dv = bentoml_status.get("data_version", "N/A") if bentoml_status else "N/A"
    fa_nf = fastapi_status.get("n_features", "N/A") if fastapi_status else "N/A"
    bm_nf = bentoml_status.get("n_features", "N/A") if bentoml_status else "N/A"

    col1, col2, col3 = st.columns(3)
    with col1:
        schema_match = fa_sv == bm_sv and fa_sv != "N/A"
        st.metric(
            label="schema_version",
            value=fa_sv[:8] if fa_sv != "N/A" else "N/A",
            delta=f"Match" if schema_match else f"BentoML={bm_sv[:8] if bm_sv != 'N/A' else 'N/A'}",
            delta_color="normal" if schema_match else "inverse",
        )
    with col2:
        data_match = fa_dv == bm_dv and fa_dv != "N/A"
        st.metric(
            label="data_version",
            value=fa_dv[:8] if fa_dv != "N/A" else "N/A",
            delta=f"Match" if data_match else f"BentoML={bm_dv[:8] if bm_dv != 'N/A' else 'N/A'}",
            delta_color="normal" if data_match else "inverse",
        )
    with col3:
        nf_match = fa_nf == bm_nf and fa_nf != "N/A"
        st.metric(
            label="n_features",
            value=str(fa_nf),
            delta=f"Match" if nf_match else f"BentoML={bm_nf}",
            delta_color="normal" if nf_match else "inverse",
        )

    if schema_match and data_match and nf_match:
        st.success("✅ FastAPI and BentoML schemas are consistent")
    else:
        st.warning("⚠️ Schema mismatch detected between FastAPI and BentoML — verify model artifacts")

    fa_features = fastapi_status.get("feature_order") if fastapi_status else None
    bm_features = bentoml_status.get("feature_order") if bentoml_status else None
    if fa_features and bm_features and fa_features != bm_features:
        st.error(f"Feature order mismatch:\n- FastAPI: {fa_features}\n- BentoML: {bm_features}")


def _render_model_lineage(fastapi_status, bentoml_status):
    st.subheader("Model Lineage")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### FastAPI")
        if fastapi_status:
            st.write(f"- **API Version**: `{fastapi_status.get('api_version', 'N/A')}`")
            st.write(f"- **Service Type**: `{fastapi_status.get('service_type', 'N/A')}`")
            st.write(f"- **Model Mode**: `{fastapi_status.get('model_mode', 'N/A')}`")
            st.write(f"- **Model Loaded**: `{fastapi_status.get('model_loaded', False)}`")
            st.write(f"- **Schema Version**: `{fastapi_status.get('schema_version', 'N/A')}`")
            st.write(f"- **Data Version**: `{fastapi_status.get('data_version', 'N/A')}`")
            st.write(f"- **N Features**: `{fastapi_status.get('n_features', 'N/A')}`")
            api_base = fastapi_status.get('api_base_suggestion', 'N/A')
            st.write(f"- **API Base Suggestion**: `{api_base}`")
        else:
            st.write("No FastAPI status data available")
    with col2:
        st.markdown("##### BentoML")
        if bentoml_status:
            st.write(f"- **API Version**: `{bentoml_status.get('api_version', 'N/A')}`")
            st.write(f"- **Service Type**: `{bentoml_status.get('service_type', 'N/A')}`")
            st.write(f"- **Model Mode**: `{bentoml_status.get('model_mode', 'N/A')}`")
            st.write(f"- **Model Loaded**: `{bentoml_status.get('model_loaded', False)}`")
            st.write(f"- **Schema Version**: `{bentoml_status.get('schema_version', 'N/A')}`")
            st.write(f"- **Data Version**: `{bentoml_status.get('data_version', 'N/A')}`")
            st.write(f"- **N Features**: `{bentoml_status.get('n_features', 'N/A')}`")
            api_base = bentoml_status.get('api_base_suggestion', 'N/A')
            st.write(f"- **API Base Suggestion**: `{api_base}`")
        else:
            st.write("No BentoML status data available")


def _render_self_check(name, status_data):
    if not status_data:
        return
    check = status_data.get("last_self_check")
    if not check:
        st.write(f"_{name}: no self-check data_")
        return
    passed = check.get("passed", False)
    ts = check.get("timestamp")
    ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "unknown"
    st.markdown(f"##### {name} Self-Check — {'✅ Passed' if passed else '❌ Failed'} at {ts_str}")
    checks = check.get("checks", [])
    if checks:
        check_rows = [
            {
                "Check": c.get("name", "?"),
                "Passed": "✅" if c.get("passed") else "❌",
                "Detail": c.get("detail", ""),
            }
            for c in checks
        ]
        st.table(check_rows)
    errors = check.get("errors", [])
    if errors:
        st.error(f"{name} Errors:\n" + "\n".join(f"- {e}" for e in errors))


def _render_error_details(fastapi_health_err, bentoml_health_err, fastapi_status_err, bentoml_status_err,
                           fastapi_status, bentoml_status):
    st.subheader("Error Details")
    any_error = False
    if fastapi_health_err:
        st.error(f"FastAPI /health: {fastapi_health_err}")
        any_error = True
    if bentoml_health_err:
        st.error(f"BentoML /health: {bentoml_health_err}")
        any_error = True
    if fastapi_status_err:
        st.error(f"FastAPI /status: {fastapi_status_err}")
        any_error = True
    if bentoml_status_err:
        st.error(f"BentoML /status: {bentoml_status_err}")
        any_error = True

    for name, s in [("FastAPI", fastapi_status), ("BentoML", bentoml_status)]:
        if s:
            sc = s.get("last_self_check", {})
            errs = sc.get("errors", [])
            if errs:
                for e in errs:
                    st.error(f"{name} self-check: {e}")
                    any_error = True
    if not any_error:
        st.success("No errors detected across all services")


def render_system_status_page():
    st.title("System Status Dashboard")
    st.write(f"FastAPI endpoint: `{API_BASE_URL}`  |  BentoML endpoint: `{BENTOML_BASE_URL}`")

    if st.button("🔄 Refresh Status"):
        fetch_fastapi_status.clear()
        fetch_fastapi_health.clear()
        fetch_bentoml_status.clear()
        fetch_bentoml_health.clear()
        st.experimental_rerun()

    st.header("Service Health")
    col1, col2 = st.columns(2)
    with col1:
        fa_health, fa_health_err = fetch_fastapi_health()
        fa_ok = _render_health_badge("FastAPI", fa_health, fa_health_err)
    with col2:
        bm_health, bm_health_err = fetch_bentoml_health()
        bm_ok = _render_health_badge("BentoML", bm_health, bm_health_err)

    st.header("Aggregated Status")
    col1, col2 = st.columns(2)
    with col1:
        fa_status, fa_status_err = fetch_fastapi_status()
        _render_status_card("FastAPI", fa_status, fa_status_err)
    with col2:
        bm_status, bm_status_err = fetch_bentoml_status()
        _render_status_card("BentoML", bm_status, bm_status_err)

    st.header("Model Lineage & Schema")
    _render_model_lineage(fa_status, bm_status)
    _compare_schemas(fa_status, bm_status)

    st.header("Self-Check Results")
    col1, col2 = st.columns(2)
    with col1:
        _render_self_check("FastAPI", fa_status)
    with col2:
        _render_self_check("BentoML", bm_status)

    _render_error_details(fa_health_err, bm_health_err, fa_status_err, bm_status_err,
                          fa_status, bm_status)


def main():
    page = st.sidebar.radio("Navigation", ["Prediction", "System Status"])
    if page == "Prediction":
        render_predict_page()
    else:
        render_system_status_page()


if __name__ == "__main__":
    main()
