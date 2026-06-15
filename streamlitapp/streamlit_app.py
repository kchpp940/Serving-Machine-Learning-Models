import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests as re
from datetime import datetime

from car_pricing.api_client import PredictionAPIClient, BentoMLAPIClient
from car_pricing.service_status import (
    compare_service_statuses,
    CONSISTENCY_OK,
    CONSISTENCY_WARNING,
    CONSISTENCY_ERROR,
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
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


@st.cache_data(show_spinner=False)
def _fetch_statuses():
    fastapi_client = PredictionAPIClient(API_BASE_URL, REQUEST_TIMEOUT)
    bentoml_client = BentoMLAPIClient(timeout=REQUEST_TIMEOUT)

    fa_health, fa_health_err = fastapi_client.health()
    fa_status, fa_status_err = fastapi_client.status()
    bm_health, bm_health_err = bentoml_client.health()
    bm_status, bm_status_err = bentoml_client.status()

    consistency_report = compare_service_statuses(
        fastapi_status=fa_status,
        bentoml_status=bm_status,
        fastapi_endpoint_error=fa_status_err,
        bentoml_endpoint_error=bm_status_err,
    )

    return fa_health, fa_health_err, fa_status, fa_status_err, bm_health, bm_health_err, bm_status, bm_status_err, consistency_report


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


def _render_consistency_banner(report):
    if report.level == CONSISTENCY_OK:
        st.success(f"### {report.summary}")
    elif report.level == CONSISTENCY_WARNING:
        st.warning(f"### {report.summary}")
    else:
        st.error(f"### {report.summary}")


def _render_lineage_table(fastapi_status, bentoml_status):
    core_fields = [
        ("api_version", "API Version"),
        ("service_type", "Service Type"),
        ("model_mode", "Model Mode"),
        ("model_loaded", "Model Loaded"),
        ("schema_version", "Schema Version"),
        ("data_version", "Data Version"),
        ("n_features", "N Features"),
        ("feature_order", "Feature Order"),
        ("api_base_suggestion", "API Base Suggestion"),
    ]
    lineage_fields = [
        ("model_name", "Model Name"),
        ("model_type", "Model Type"),
        ("source_run_id", "Source Run ID"),
        ("model_artifact_hash", "Artifact Hash"),
        ("primary_metric", "Primary Metric"),
        ("candidate_summary_path", "Candidate Summary Path"),
    ]
    rows = []

    for field, label in core_fields:
        fa_val = fastapi_status.get(field, "N/A") if fastapi_status else "N/A"
        bm_val = bentoml_status.get(field, "N/A") if bentoml_status else "N/A"
        fa_str = str(fa_val) if not isinstance(fa_val, list) else str(fa_val)
        bm_str = str(bm_val) if not isinstance(bm_val, list) else str(bm_val)
        match = "✅" if fa_val == bm_val else "❌"
        rows.append({"Category": "Core", "Field": label, "FastAPI": fa_str, "BentoML": bm_str, "Match": match})

    fa_lineage = fastapi_status.get("lineage") or {} if fastapi_status else {}
    bm_lineage = bentoml_status.get("lineage") or {} if bentoml_status else {}

    for field, label in lineage_fields:
        fa_val = fa_lineage.get(field, "N/A") if fa_lineage else "N/A"
        bm_val = bm_lineage.get(field, "N/A") if bm_lineage else "N/A"
        fa_str = str(fa_val)
        bm_str = str(bm_val)
        if field == "model_artifact_hash" and fa_val and fa_val != "N/A":
            fa_str = fa_str[:16] + "..." if len(fa_str) > 16 else fa_str
        if field == "model_artifact_hash" and bm_val and bm_val != "N/A":
            bm_str = bm_str[:16] + "..." if len(bm_str) > 16 else bm_str
        match = "✅" if fa_val == bm_val else "❌"
        rows.append({"Category": "Lineage", "Field": label, "FastAPI": fa_str, "BentoML": bm_str, "Match": match})

    st.table(rows)


def _render_diffs_table(report):
    if not report.diffs:
        st.success("No schema/lineage differences detected")
        return
    rows = []
    for d in report.diffs:
        sev_icon = "❌" if d.severity == CONSISTENCY_ERROR else "⚠️"
        fa_val = d.fastapi_value
        bm_val = d.bentoml_value
        fa_str = str(fa_val) if not isinstance(fa_val, list) else str(fa_val)
        bm_str = str(bm_val) if not isinstance(bm_val, list) else str(bm_val)
        if d.field == "lineage.model_artifact_hash":
            if fa_val:
                fa_str = fa_val[:16] + "..." if len(fa_val) > 16 else fa_val
            if bm_val:
                bm_str = bm_val[:16] + "..." if len(bm_val) > 16 else bm_val
        rows.append({
            "Severity": sev_icon,
            "Field": d.field,
            "FastAPI": fa_str,
            "BentoML": bm_str,
        })
    st.table(rows)


def _render_all_errors(report, fa_health_err, bm_health_err):
    all_errors = []

    if fa_health_err:
        all_errors.append({"Service": "FastAPI", "Type": "/health endpoint", "Detail": fa_health_err})
    if bm_health_err:
        all_errors.append({"Service": "BentoML", "Type": "/health endpoint", "Detail": bm_health_err})

    for se in report.service_errors:
        svc = se.service_type
        for e in se.endpoint_errors:
            all_errors.append({"Service": svc.capitalize(), "Type": "/status endpoint", "Detail": e})
        for e in se.self_check_errors:
            all_errors.append({"Service": svc.capitalize(), "Type": "Self-check", "Detail": e})

    if all_errors:
        st.error(f"### Error Details ({len(all_errors)} issue(s))")
        st.table(all_errors)
    else:
        st.success("### Error Details — No errors detected across all services")


def render_system_status_page():
    st.title("System Status Dashboard")
    bentoml_base = os.environ.get("BENTOML_BASE_URL", "http://localhost:3000")
    st.write(f"FastAPI endpoint: `{API_BASE_URL}`  |  BentoML endpoint: `{bentoml_base}`")

    if st.button("🔄 Refresh Status"):
        _fetch_statuses.clear()
        st.experimental_rerun()

    fa_health, fa_health_err, fa_status, fa_status_err, bm_health, bm_health_err, bm_status, bm_status_err, report = _fetch_statuses()

    st.header("Consistency Summary")
    _render_consistency_banner(report)

    st.header("Service Health")
    col1, col2 = st.columns(2)
    with col1:
        _render_health_badge("FastAPI", fa_health, fa_health_err)
    with col2:
        _render_health_badge("BentoML", bm_health, bm_health_err)

    st.header("Model Lineage Comparison")
    _render_lineage_table(fa_status, bm_status)

    st.header("Schema & Version Diffs")
    _render_diffs_table(report)

    st.header("Aggregated Status (Raw)")
    col1, col2 = st.columns(2)
    with col1:
        _render_status_card("FastAPI", fa_status, fa_status_err)
    with col2:
        _render_status_card("BentoML", bm_status, bm_status_err)

    _render_all_errors(report, fa_health_err, bm_health_err)


def main():
    page = st.sidebar.radio("Navigation", ["Prediction", "System Status"])
    if page == "Prediction":
        render_predict_page()
    else:
        render_system_status_page()


if __name__ == "__main__":
    main()
