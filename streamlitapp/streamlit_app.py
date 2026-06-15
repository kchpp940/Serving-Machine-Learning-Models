import streamlit as st
import pandas as pd
import uuid
from copy import deepcopy

from api_client import (
    API_BASE_URL,
    SchemaResponse,
    BatchTransportResponse,
    BatchRowResult,
    fetch_schema,
    predict_batch,
    get_default_values,
    get_display_name,
    validate_values,
    format_value,
    get_api_base_url,
    build_fallback_schema,
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
    if "field_warnings" not in st.session_state:
        st.session_state.field_warnings = {}
    if "server_field_errors" not in st.session_state:
        st.session_state.server_field_errors = {}
    if "global_error" not in st.session_state:
        st.session_state.global_error = None
    if "global_info" not in st.session_state:
        st.session_state.global_info = None
    if "schema" not in st.session_state:
        st.session_state.schema = schema

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
    for key in ("scenario_names", "scenario_values", "results", "field_warnings", "server_field_errors"):
        st.session_state[key].pop(scenario_id, None)


def copy_scenario(schema: dict, scenario_id: str):
    src_name = st.session_state.scenario_names.get(scenario_id, "Scenario")
    add_scenario(schema, name=f"{src_name} (Copy)", source_id=scenario_id)


def render_scenario_form(scenario_id: str, schema: dict, expanded: bool = True):
    name = st.session_state.scenario_names.get(scenario_id, "")
    values = st.session_state.scenario_values.get(scenario_id, {})
    warnings = st.session_state.field_warnings.get(scenario_id, {})
    server_errors = st.session_state.server_field_errors.get(scenario_id, {})

    feature_order = schema["feature_order"]
    numeric_features = set(schema["numeric_features"])
    categorical_features = set(schema["categorical_features"])
    categorical_options = schema["categorical_options"]

    display_name = name or f"Scenario {scenario_id}"
    result = st.session_state.results.get(scenario_id)
    result_label = ""
    if result and result.get("prediction") is not None:
        result_label = f" · **${result['prediction']:,.2f}**"
    elif result and result.get("error"):
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
                label = get_display_name(field, schema)
                has_warning = field in warnings
                has_error = field in server_errors
                warning_msg = warnings.get(field, "")
                error_msg = server_errors.get(field, "")

                if has_error:
                    st.markdown(f"<span style='color:#ff6b6b'>❌ {label} — {error_msg}</span>", unsafe_allow_html=True)
                elif has_warning:
                    st.markdown(f"<span style='color:#ffc107'>⚠️ {label} — {warning_msg}</span>", unsafe_allow_html=True)

                current_val = values.get(field)
                if field in numeric_features:
                    step = 1.0 if field in ("enginesize", "curbweight", "horsepower", "highwaympg", "citympg") else 0.1
                    val = st.number_input(
                        label if not (has_warning or has_error) else " ",
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
                            label if not (has_warning or has_error) else " ",
                            range(len(display_labels)),
                            index=current_idx,
                            format_func=lambda i, dl=display_labels: dl[i],
                            key=f"cat_{scenario_id}_{field}",
                        )
                        st.session_state.scenario_values[scenario_id][field] = form_values[sel_idx]
                    else:
                        val = st.text_input(
                            label if not (has_warning or has_error) else " ",
                            value=str(current_val) if current_val is not None else "",
                            key=f"textcat_{scenario_id}_{field}",
                        )
                        st.session_state.scenario_values[scenario_id][field] = val
                else:
                    val = st.text_input(
                        label if not (has_warning or has_error) else " ",
                        value=str(current_val) if current_val is not None else "",
                        key=f"text_{scenario_id}_{field}",
                    )
                    st.session_state.scenario_values[scenario_id][field] = val

        if result:
            st.markdown("---")
            if result.get("prediction") is not None:
                st.success(f"✅ Predicted Price: **${result['prediction']:,.2f}**")
            if result.get("error"):
                err = result["error"]
                fe = result.get("field_errors")
                if fe:
                    field_list = ", ".join(f"{get_display_name(k, schema)}: {v}" for k, v in fe.items())
                    st.error(f"❌ {err} — {field_list}")
                else:
                    st.error(f"❌ {err}")


def find_differing_fields(result_ids: list[str], schema: dict) -> set[str]:
    if len(result_ids) < 2:
        return set()
    differing: set[str] = set()
    ref_id = result_ids[0]
    ref_values = st.session_state.scenario_values.get(ref_id, {})
    for sid in result_ids[1:]:
        vals = st.session_state.scenario_values.get(sid, {})
        for field in schema["feature_order"]:
            if vals.get(field) != ref_values.get(field):
                differing.add(field)
    return differing


def render_comparison_table(schema: dict):
    all_result_ids = [sid for sid in st.session_state.scenario_ids if sid in st.session_state.results]
    if not all_result_ids:
        return

    results_with_data = []
    for sid in all_result_ids:
        r = st.session_state.results[sid]
        results_with_data.append((sid, r))

    successful = [(sid, r) for sid, r in results_with_data if r.get("prediction") is not None]
    successful.sort(key=lambda x: x[1]["prediction"])

    all_ids = [sid for sid, _ in results_with_data]
    differing_fields = find_differing_fields(all_ids, schema)

    st.markdown("## 📊 Comparison Results")

    failed = [(sid, r) for sid, r in results_with_data if r.get("error")]
    if failed:
        st.warning(f"{len(failed)} scenario(s) returned errors. Valid scenarios still predicted successfully.")

    if successful:
        table_data = []
        for rank, (sid, r) in enumerate(successful, start=1):
            row = {
                "Rank": rank,
                "Scenario": st.session_state.scenario_names.get(sid, sid),
                "Predicted Price": f"${r['prediction']:,.2f}",
            }
            for field in schema["feature_order"]:
                val = st.session_state.scenario_values.get(sid, {}).get(field, "")
                display_val = format_value(field, val, schema)
                if field in differing_fields:
                    row[get_display_name(field, schema) + " 🔺"] = display_val
                else:
                    row[get_display_name(field, schema)] = display_val
            table_data.append(row)

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if differing_fields:
            st.info(
                "🔺 Differing fields highlighted. "
                f"{len(differing_fields)} field(s) vary across scenarios: "
                + ", ".join(get_display_name(f, schema) for f in differing_fields)
            )

        if failed:
            st.markdown("### ⚠️ Failed Scenarios")
            error_data = []
            for sid, r in failed:
                fe = r.get("field_errors")
                error_detail = r["error"]
                if fe:
                    field_list = "; ".join(f"{get_display_name(k, schema)}: {v}" for k, v in fe.items())
                    error_detail = f"{error_detail} — {field_list}"
                error_data.append({
                    "Scenario": st.session_state.scenario_names.get(sid, sid),
                    "Error": error_detail,
                })
            st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)
    else:
        if failed:
            st.error("No successful predictions. All scenarios have errors — see details below.")
            error_data = []
            for sid, r in failed:
                fe = r.get("field_errors")
                error_detail = r["error"]
                if fe:
                    field_list = "; ".join(f"{get_display_name(k, schema)}: {v}" for k, v in fe.items())
                    error_detail = f"{error_detail} — {field_list}"
                error_data.append({
                    "Scenario": st.session_state.scenario_names.get(sid, sid),
                    "Error": error_detail,
                })
            st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)


def run_client_side_validation(schema: dict) -> bool:
    st.session_state.field_warnings = {}
    has_any_warning = False
    for sid in st.session_state.scenario_ids:
        values = st.session_state.scenario_values.get(sid, {})
        name = st.session_state.scenario_names.get(sid, "")
        warns = {}
        if not name:
            warns["__name__"] = "Scenario name is required"
            has_any_warning = True
        field_warns = validate_values(values, schema)
        if field_warns:
            warns.update(field_warns)
            has_any_warning = True
        if warns:
            st.session_state.field_warnings[sid] = warns
    return has_any_warning


def run_prediction_batch(schema: dict):
    st.session_state.global_error = None
    st.session_state.global_info = None
    st.session_state.server_field_errors = {}

    has_warnings = run_client_side_validation(schema)

    rows = []
    row_ids = []
    for sid in st.session_state.scenario_ids:
        values = st.session_state.scenario_values.get(sid, {})
        rows.append(deepcopy(values))
        row_ids.append(sid)

    with st.spinner(f"Running batch prediction for {len(rows)} scenario(s)..."):
        batch_response: BatchTransportResponse = predict_batch(rows=rows, row_ids=row_ids)

    if batch_response.transport_error:
        st.session_state.global_error = batch_response.transport_error
        if has_warnings:
            st.session_state.global_info = (
                "Client-side validation also found issues in some scenarios, "
                "but the primary issue is the service connection."
            )
        return

    for row_result in batch_response.results:
        sid = row_result.row_id
        if sid is None:
            continue
        result_data = {
            "prediction": row_result.prediction,
            "error": row_result.error,
            "field_errors": row_result.field_errors,
        }
        st.session_state.results[sid] = result_data
        if row_result.field_errors:
            st.session_state.server_field_errors[sid] = dict(row_result.field_errors)

    success = batch_response.success_count
    errors = batch_response.error_count
    total = batch_response.total_count

    if success > 0 and errors == 0:
        st.session_state.global_info = f"✅ All {success} prediction(s) succeeded."
    elif success > 0 and errors > 0:
        st.session_state.global_info = (
            f"✅ {success} succeeded, ⚠️ {errors} failed. "
            "Invalid scenarios are shown with errors; valid ones still predicted."
        )
    elif errors == total and total > 0:
        st.session_state.global_error = (
            f"⚠️ All {errors} scenario(s) failed. "
            "Check the field errors in each scenario and try again."
        )

    if has_warnings and success > 0:
        if not st.session_state.global_info:
            st.session_state.global_info = (
                "Some scenarios have client-side warnings but were still submitted. "
                "The server validated and processed valid rows independently."
            )


def main():
    st.title("🚗 Car Price — Scenario Comparison")
    st.markdown(
        "Compare multiple vehicle configurations side-by-side. "
        "Add scenarios, copy existing ones, tweak a few fields, and predict all at once. "
        "Valid scenarios always predict even if some have errors."
    )

    with st.spinner("Loading feature schema from prediction service..."):
        schema_resp: SchemaResponse = fetch_schema()

    if schema_resp.error and schema_resp.using_fallback:
        st.warning(
            f"{schema_resp.error} Using built-in fallback schema. "
            "The form may not match the server's expectations until the service is reachable."
        )
        schema = schema_resp.schema
    else:
        base_url = get_api_base_url()
        feat_count = len(schema_resp.schema.get("feature_order", [])) if schema_resp.schema else 0
        st.success(f"✅ Schema loaded from `{base_url}` ({feat_count} features)")
        schema = schema_resp.schema

    if schema is None:
        schema = build_fallback_schema()

    init_session_state(schema)

    if st.session_state.global_error:
        st.error(st.session_state.global_error)
    if st.session_state.global_info:
        st.info(st.session_state.global_info)

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
