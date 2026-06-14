import sys
import os
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from car_pricing.api_client import (
    CarPriceAPIClient,
    SchemaInfo,
    BatchPredictionResponse,
    APIConnectionError,
    APITimeoutError,
    APIHTTPError,
    APIClientError,
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

FALLBACK_SCHEMA_DICT = {
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


def get_client() -> CarPriceAPIClient:
    return CarPriceAPIClient()


@st.cache_data(show_spinner=False)
def fetch_schema_cached():
    client = get_client()
    try:
        schema = client.get_schema()
        return schema, None
    except APIConnectionError:
        return None, "Unable to connect to the prediction service to fetch schema."
    except APITimeoutError:
        return None, "Schema request timed out."
    except APIHTTPError as e:
        return None, f"Server returned error when fetching schema: {e.detail or str(e)}"
    except APIClientError as e:
        return None, f"Unexpected error fetching schema: {str(e)}"


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
                selected_idx = st.selectbox(label, range(len(display_labels)), format_func=lambda i: display_labels[i])
                inputs[field] = options[selected_idx]["form_value"]
        else:
            inputs[field] = st.text_input(label)
    return inputs


def df_to_records(df: pd.DataFrame, schema: SchemaInfo) -> list:
    feature_order = schema.feature_order
    records = []
    for _, row in df.iterrows():
        record = {}
        for f in feature_order:
            val = row[f]
            if isinstance(val, pd.Timestamp):
                record[f] = str(val)
            elif pd.isna(val):
                record[f] = None
            else:
                record[f] = val.item() if hasattr(val, "item") else val
        records.append(record)
    return records


def render_single_prediction(schema: SchemaInfo):
    st.header("Input Car Details")
    names = st.text_input("Name of Car")

    inputs = build_form(schema)

    if st.button("Predict Price"):
        if not names:
            st.error("Please enter the name of the car.")
            return

        values = {}
        for field in schema.feature_order:
            values[field] = inputs[field]

        client = get_client()
        try:
            prediction = client.predict(values)
            st.success(f"The Price of the {names} is {prediction:.2f}$")
        except APIConnectionError:
            st.error("Unable to connect to the prediction service. Please check that the API server is running.")
        except APITimeoutError:
            st.error("The request to the prediction service timed out. Please try again later.")
        except APIHTTPError as e:
            st.error(f"Server returned an error ({e.status_code}): {e.detail or str(e)}")
        except APIClientError as e:
            st.error(f"An unexpected error occurred: {str(e)}")


def render_batch_prediction(schema: SchemaInfo):
    st.header("Batch Prediction (CSV Upload)")

    st.info(
        "Upload a CSV file with the following columns: "
        + ", ".join(f"`{f}`" for f in schema.feature_order)
    )

    with st.expander("Show expected schema details"):
        st.markdown("**Required columns:**")
        for f in schema.feature_order:
            label = FIELD_DISPLAY_NAMES.get(f, f)
            if f in schema.numeric_features:
                st.markdown(f"- `{f}` ({label}): numeric value")
            elif f in schema.categorical_features:
                opts = schema.categorical_options.get(f, [])
                valid = ", ".join(f"`{o['form_value']}`" for o in opts)
                st.markdown(f"- `{f}` ({label}): categorical, valid values are {valid}")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV file: {str(e)}")
            return

        st.write(f"**Loaded {len(df)} rows, {len(df.columns)} columns**")

        with st.expander("Preview uploaded data"):
            st.dataframe(df.head(min(10, len(df))))

        client = get_client()
        missing_cols, extra_cols = client.validate_csv_columns(list(df.columns), schema)

        if missing_cols:
            st.error(f"CSV is missing required columns: {', '.join(f'`{c}`' for c in missing_cols)}")
            return

        if extra_cols:
            st.warning(f"CSV has extra columns that will be ignored: {', '.join(f'`{c}`' for c in extra_cols)}")

        records = df_to_records(df, schema)
        row_errors = client.validate_csv_rows(records, schema)
        valid_count = len(df) - len(row_errors)

        if row_errors:
            st.error(f"Found {len(row_errors)} invalid row(s) out of {len(df)}. Valid rows: {valid_count}")
            with st.expander("Show row validation errors"):
                for err in row_errors:
                    st.markdown(f"**Row {err['row_index']}:**")
                    for msg in err["errors"]:
                        st.markdown(f"- {msg}")
        else:
            st.success(f"All {len(df)} rows passed validation.")

        if valid_count > 0:
            if st.button("Run Batch Prediction"):
                try:
                    with st.spinner("Sending batch prediction request..."):
                        batch_resp = client.predict_batch(records)

                    st.success(
                        f"Batch prediction complete: "
                        f"{batch_resp.valid_count} valid, "
                        f"{batch_resp.invalid_count} invalid "
                        f"(total {batch_resp.total_records})"
                    )

                    if batch_resp.results:
                        result_rows = batch_resp.to_dataframe_rows()
                        result_df = pd.DataFrame(result_rows)

                        st.subheader("Prediction Results")
                        valid_df = result_df[result_df["error"].isna()].drop(columns=["error"])
                        invalid_df = result_df[result_df["error"].notna()]

                        if not valid_df.empty:
                            st.markdown("**Successful predictions:**")
                            st.dataframe(valid_df, use_container_width=True)

                            csv_buffer = io.StringIO()
                            valid_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="Download predictions as CSV",
                                data=csv_buffer.getvalue(),
                                file_name="predictions.csv",
                                mime="text/csv",
                            )

                        if not invalid_df.empty:
                            st.markdown("**Failed rows:**")
                            st.dataframe(invalid_df, use_container_width=True)

                except APIConnectionError:
                    st.error("Unable to connect to the prediction service. Please check that the API server is running.")
                except APITimeoutError:
                    st.error("The batch prediction request timed out. Please try again with fewer rows or later.")
                except APIHTTPError as e:
                    st.error(f"Server returned an error ({e.status_code}): {e.detail or str(e)}")
                except APIClientError as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
        else:
            st.warning("No valid rows to predict. Please fix the CSV errors and try again.")


def main():
    st.title("Car Price Prediction Web App")

    st.write("""
    ## About

    **This Streamlit App utilizes a Machine Learning model served as an API to predict the price of a car based on certain features.**

    """)

    schema, schema_error = fetch_schema_cached()
    using_fallback = False
    if schema_error is not None:
        st.warning(f"{schema_error} Using default schema. The form may not match the server's expectations.")
        schema = SchemaInfo.from_dict(FALLBACK_SCHEMA_DICT)
        using_fallback = True
    else:
        st.success(f"Loaded schema from API ({len(schema.feature_order)} features)")

    mode = st.radio("Prediction Mode", ["Single Prediction", "Batch Prediction (CSV Upload)"], index=0)

    if mode == "Single Prediction":
        render_single_prediction(schema)
    else:
        render_batch_prediction(schema)


if __name__ == "__main__":
    main()
