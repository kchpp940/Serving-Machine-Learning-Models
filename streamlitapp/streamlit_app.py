import streamlit as st
import pandas as pd
import uuid
from copy import deepcopy

from api_client import (
    API_BASE_URL,
    DEFAULT_SCHEMA,
    PredictionResult,
    fetch_schema,
    predict_batch,
    get_default_values,
    get_display_name,
    validate_values,
)

st.set_page_config(page_title="Car Price Comparison", layout="wide")


def init_session_state(schema: dict):
    if "scenario_ids" not in st.session_state:
        st.session_state.scenario_ids = []
    if "scenario_names" not in st.session_state:
        st.session_state.scenario_names = {}
    if "scenario_values" not in st.session_state:
        st.session_state.scenario_values = {}
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "field_errors" not in st.session_state:
        st.session_state.field_errors = {}
    if "global_error" not in st.session_state:
        st.session_state.global_error = None

    if not st.session_state.scenario_ids:
        add_scenario(schema, name="Scenario 1")


def add_scenario(schema: dict, name: str | None = None, source_id: str | None = None):
    new_id = str(uuid.uuid4())[:8]
    if name is None:
        idx = len(st.session_state.scenario_ids) + 1
        name = f"Scenario {idx}"
    if source_id is not None and source_id in st.session_state.scenario_values:
        values = deepcopy(st.session_state.scenario_values[source_id])
    else:
        values = get_default_values(schema)
    st.session_state.scenario_ids.append(new_id)
    st.session_state.scenario_names[new_id] = name
    st.session_state.scenario_values[new_id] = values
    return new_id


def remove_scenario(scenario_id: str):
    if scenario_id in st.session_state.scenario_ids:
        st.session_state.scenario_ids.remove(scenario_id)
    for key in ("scenario_names", "scenario_values", "results", "field_errors"):
        st.session_state[key].pop(scenario_id, None)


def copy_scenario(schema: dict, scenario_id: str):
    src_name = st.session_state.scenario_names.get(scenario_id, "Scenario")
    add_scenario(schema, name=f"{src_name} (Copy)", source_id=scenario_id)


def render_scenario_form(scenario_id: str, schema: dict, expanded: bool = True):
    name = st.session_state.scenario_names.get(scenario_id, "")
    values = st.session_state.scenario_values.get(scenario_id, {})
    errors = st.session_state.field_errors.get(scenario_id, {})

    feature_order = schema["feature_order"]
    numeric_features = set(schema["numeric_features"])
    categorical_features = set(schema["categorical_features"])
    categorical_options = schema["categorical_options"]

    display_name = name or f"Scenario {scenario_id}"
    result = st.session_state.results.get(scenario_id)
    result_label = ""
    if result and result.prediction is not None:
        result_label = f" · **${result.prediction:,.2f}**"
    elif result and result.error:
        result_label = " · ⚠️ Error"

    with st.expander(f"📋 {display_name}{result_label}", expanded=expanded):
        col_name, col_actions = st.columns([3, 1])
        with col_name:
            new_name = st.text_input(
                "Scenario Name",
                value=name,
                key=f"name_{scenario_id}",
                label_visibility="collapsed",
            )
            if new_name != name:
                st.session_state.scenario_names[scenario_id] = new_name
        with col_actions:
            btn_cols = st.columns(2)
            with btn_cols[0]:
                if st.button("📑 Copy", key=f"copy_{scenario_id}", use_container_width=True):
                    copy_scenario(schema, scenario_id)
                    st.rerun()
            with btn_cols[1]:
                if st.button("🗑️ Remove", key=f"remove_{scenario_id}", use_container_width=True):
                    if len(st.session_state.scenario_ids) <= 1:
                        st.warning("At least one scenario must remain.")
                    else:
                        remove_scenario(scenario_id)
                        st.rerun()

        st.markdown("---")

        n_cols = 2
        cols = st.columns(n_cols)
        for i, field in enumerate(feature_order):
            with cols[i % n_cols]:
                label = get_display_name(field)
                has_error = field in errors
                error_msg = errors.get(field, "")
                if has_error:
                    st.markdown(f"<span style='color:#ff6b6b'>⚠️ {label}</span>", unsafe_allow_html=True)
                current_val = values.get(field)
                if field in numeric_features:
                    step = 1.0 if field in ("enginesize", "curbweight", "horsepower", "highwaympg", "citympg") else 0.1
                    val = st.number_input(
                        label if not has_error else f"{label} — {error_msg}",
                        value=float(current_val) if current_val is not None else 0.0,
                        step=step,
                        key=f"num_{scenario_id}_{field}",
                    )
                    st.session_state.scenario_values[scenario_id][field] = val
                elif field in categorical_features:
                    opts = categorical_options.get(field, [])
                    if opts:
                        display_labels = [opt["display"] for opt in opts]
                        form_values = [opt["form_value"] for opt in opts]
                        current_idx = 0
                        if current_val in form_values:
                            current_idx = form_values.index(current_val)
                        sel_idx = st.selectbox(
                            label if not has_error else f"{label} — {error_msg}",
                            range(len(display_labels)),
                            index=current_idx,
                            format_func=lambda i, dl=display_labels: dl[i],
                            key=f"cat_{scenario_id}_{field}",
                        )
                        st.session_state.scenario_values[scenario_id][field] = form_values[sel_idx]
                    else:
                        val = st.text_input(
                            label if not has_error else f"{label} — {error_msg}",
                            value=str(current_val) if current_val is not None else "",
                            key=f"textcat_{scenario_id}_{field}",
                        )
                        st.session_state.scenario_values[scenario_id][field] = val
                else:
                    val = st.text_input(
                        label if not has_error else f"{label} — {error_msg}",
                        value=str(current_val) if current_val is not None else "",
                        key=f"text_{scenario_id}_{field}",
                    )
                    st.session_state.scenario_values[scenario_id][field] = val

        if result:
            st.markdown("---")
            if result.prediction is not None:
                st.success(f"✅ Predicted Price: **${result.prediction:,.2f}**")
            if result.error:
                st.error(f"❌ {result.error}")


def find_differing_fields(results: list[PredictionResult], schema: dict) -> set[str]:
    if len(results) < 2:
        return set()
    differing: set[str] = set()
    ref_id = results[0].scenario_id
    ref_values = st.session_state.scenario_values.get(ref_id, {})
    for r in results[1:]:
        vals = st.session_state.scenario_values.get(r.scenario_id, {})
        for field in schema["feature_order"]:
            if vals.get(field) != ref_values.get(field):
                differing.add(field)
    return differing


def format_value_for_display(field: str, value, schema: dict) -> str:
    categorical_options = schema.get("categorical_options", {})
    opts = categorical_options.get(field, [])
    if opts:
        for opt in opts:
            if opt["form_value"] == value:
                return opt["display"]
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value)}"
        return f"{value:.2f}"
    return str(value)


def render_comparison_table(schema: dict):
    all_results: list[PredictionResult] = []
    for sid in st.session_state.scenario_ids:
        r = st.session_state.results.get(sid)
        if r:
            all_results.append(r)

    if not all_results:
        return

    successful = [r for r in all_results if r.prediction is not None]
    successful.sort(key=lambda r: r.prediction)

    differing_fields = find_differing_fields(all_results, schema)

    st.markdown("## 📊 Comparison Results")

    if any(r.error for r in all_results):
        error_count = sum(1 for r in all_results if r.error)
        st.warning(f"{error_count} scenario(s) returned errors. Scroll down for details.")

    if successful:
        table_data = []
        for rank, r in enumerate(successful, start=1):
            row = {
                "Rank": rank,
                "Scenario": st.session_state.scenario_names.get(r.scenario_id, r.scenario_id),
                "Predicted Price": f"${r.prediction:,.2f}",
            }
            for field in schema["feature_order"]:
                val = st.session_state.scenario_values.get(r.scenario_id, {}).get(field, "")
                display_val = format_value_for_display(field, val, schema)
                if field in differing_fields:
                    row[get_display_name(field) + " 🔺"] = display_val
                else:
                    row[get_display_name(field)] = display_val
            table_data.append(row)

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if differing_fields:
            st.info(
                "🔺 Differing fields highlighted. "
                f"{len(differing_fields)} field(s) vary across scenarios: "
                + ", ".join(get_display_name(f) for f in differing_fields)
            )

        failed = [r for r in all_results if r.error]
        if failed:
            st.markdown("### ⚠️ Failed Scenarios")
            error_data = []
            for r in failed:
                error_data.append({
                    "Scenario": st.session_state.scenario_names.get(r.scenario_id, r.scenario_id),
                    "Error": r.error,
                })
            st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)
    else:
        st.error("No successful predictions to display.")


def run_prediction_batch(schema: dict):
    st.session_state.global_error = None
    st.session_state.field_errors = {}

    has_any_error = False
    batch_inputs: list[PredictionResult] = []

    for sid in st.session_state.scenario_ids:
        values = st.session_state.scenario_values.get(sid, {})
        name = st.session_state.scenario_names.get(sid, "")
        if not name:
            st.session_state.field_errors.setdefault(sid, {})["__name__"] = "Scenario name is required"
            has_any_error = True

        field_errs = validate_values(values, schema)
        if field_errs:
            st.session_state.field_errors[sid] = field_errs
            has_any_error = True

        batch_inputs.append(PredictionResult(
            scenario_id=sid,
            name=name,
            values=deepcopy(values),
        ))

    if has_any_error:
        st.session_state.global_error = "Some scenarios have invalid fields. Please fix them and try again."
        return

    with st.spinner(f"Running predictions for {len(batch_inputs)} scenario(s)..."):
        results = predict_batch(batch_inputs)

    for r in results:
        st.session_state.results[r.scenario_id] = r

    has_success = any(r.prediction is not None for r in results)
    has_failure = any(r.error for r in results)

    if has_success and not has_failure:
        st.session_state.global_error = None
    elif has_failure and not has_success:
        st.session_state.global_error = "All prediction requests failed. Check the service status and try again."
    else:
        st.session_state.global_error = "Some predictions succeeded while others failed. See details below."


def main():
    st.title("🚗 Car Price — Scenario Comparison")
    st.markdown(
        "Compare multiple vehicle configurations side-by-side. "
        "Add scenarios, copy existing ones, tweak a few fields, and predict all at once."
    )

    with st.spinner("Loading feature schema from prediction service..."):
        schema, schema_error = fetch_schema()

    if schema_error is not None:
        st.warning(
            f"{schema_error} Using built-in fallback schema. "
            "The form may not match the server's expectations until the service is reachable."
        )
        schema = DEFAULT_SCHEMA
    else:
        st.success(f"✅ Schema loaded from `{API_BASE_URL}` ({len(schema['feature_order'])} features)")

    init_session_state(schema)

    if st.session_state.global_error:
        st.error(st.session_state.global_error)

    st.markdown("---")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        if st.button("➕ Add Scenario", use_container_width=True):
            add_scenario(schema)
            st.rerun()
    with ctrl_col2:
        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with ctrl_col3:
        st.markdown(
            f"<div style='text-align:right; padding-top:8px; color:#666'>"
            f"{len(st.session_state.scenario_ids)} scenario(s) configured</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    for i, sid in enumerate(st.session_state.scenario_ids):
        render_scenario_form(sid, schema, expanded=(i < 2))

    st.markdown("---")

    pred_col, _ = st.columns([1, 3])
    with pred_col:
        if st.button("🚀 Predict All Scenarios", type="primary", use_container_width=True):
            run_prediction_batch(schema)
            st.rerun()

    render_comparison_table(schema)


if __name__ == "__main__":
    main()
