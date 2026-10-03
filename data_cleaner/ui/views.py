"""The main sections of the Streamlit app: data, recommendations, clean, combine,
quarantine, plot and code. Each ``render_*`` function draws one section."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from data_cleaner.core.dataset import Dataset
from data_cleaner.core.operations import OPERATIONS
from data_cleaner.core.plotting import (
    DATA,
    DEFAULT_MAX_POINTS,
    MARKER_SYMBOLS,
    PLOT_KINDS,
    STYLE,
    PlotArgument,
    build_figure,
    fixed_key,
    plot_code,
)
from data_cleaner.core.recommendations import Recommendation, apply_all_recommendations, recommend
from data_cleaner.core.workspace import Workspace

from .forms import FORMS
from .state import run_action

MAX_EDITOR_ROWS = 100_000  # the editable grid shows at most this many rows
_PRIORITY_BADGES = {1: ":red[High]", 2: ":orange[Medium]", 3: ":gray[Low]"}


def _title(operation_name: str) -> str:
    return operation_name.replace("_", " ").capitalize()


# ------------------------------------------------------------------------------ data


def render_data(dataset: Dataset) -> None:
    current = dataset.current
    rows, columns = current.shape
    metrics = st.columns(4)
    metrics[0].metric("Rows", f"{rows:,}")
    metrics[1].metric("Columns", f"{columns:,}")
    metrics[2].metric("Null cells", f"{int(current.isna().sum().sum()):,}")
    metrics[3].metric("Duplicate rows", f"{int(current.duplicated().sum()):,}")

    view = st.segmented_control("Show", ["Current (editable)", "Original"], default="Current (editable)", key="data_view") or "Current (editable)"
    if view == "Original":
        st.caption("The untouched data as loaded. It never changes.")
        st.dataframe(dataset.original, alt=f"Original {dataset.name} data")
    else:
        _render_editable_table(dataset)

    with st.expander("Column profile"):
        profile = pd.DataFrame(
            {
                "dtype": current.dtypes.astype(str),
                "nulls": current.isna().sum(),
                "null %": (current.isna().mean() * 100).round(1),
                "unique": current.nunique(),
                # Mixed types in one column break Arrow serialization, so show examples as text.
                "example": current.apply(lambda column: str(column.dropna().iloc[0]) if column.notna().any() else ""),
            }
        )
        st.dataframe(profile, alt="Per-column data type, null count and example value")


def _render_editable_table(dataset: Dataset) -> None:
    """Editable grid: edits, row deletions and additions are committed as cleaning steps."""
    current = dataset.current
    discards = st.session_state.setdefault("editor_discards", 0)
    editor_key = f"editor|{dataset.name}|{dataset.cursor}|{discards}"
    shown = current.head(MAX_EDITOR_ROWS)
    if len(current) > MAX_EDITOR_ROWS:
        st.caption(f"Showing the first {MAX_EDITOR_ROWS:,} of {len(current):,} rows for editing.")
    st.caption("Click a cell to edit it, tick the row checkbox and press Delete to remove rows, or add a row at the bottom.")
    st.data_editor(shown, num_rows="dynamic", key=editor_key, alt=f"Editable {dataset.name} data")

    changes = st.session_state.get(editor_key, {})
    edited_rows = changes.get("edited_rows", {})
    added_rows = changes.get("added_rows", [])
    deleted_rows = changes.get("deleted_rows", [])
    pending = sum(len(cells) for cells in edited_rows.values()) + len(added_rows) + len(deleted_rows)
    if not pending:
        return
    st.info(f"{pending} pending change(s): {len(edited_rows)} row(s) edited, {len(deleted_rows)} deleted, {len(added_rows)} added.")
    commit_column, discard_column, _ = st.columns([1, 1, 4])
    if commit_column.button("Commit changes", type="primary", icon=":material/check:"):
        run_action(
            lambda: dataset.apply_editor_changes(edited_rows, added_rows, deleted_rows),
            "Manual edits recorded as steps.",
        )
    if discard_column.button("Discard", icon=":material/undo:"):
        st.session_state["editor_discards"] = discards + 1
        st.rerun()


# ------------------------------------------------------------------- recommendations


def render_recommendations(dataset: Dataset) -> None:
    recommendations = _cached_recommendations(dataset)
    if not recommendations:
        st.success("No cleaning steps recommended - the data looks tidy.", icon=":material/task_alt:")
        return
    text_column, button_column = st.columns([4, 1], vertical_alignment="center")
    text_column.caption(
        f"{len(recommendations)} suggestion(s), most important first. Each applied suggestion is a separate step you can undo."
    )
    if button_column.button(f"Apply all ({len(recommendations)})", type="primary", icon=":material/done_all:", width="stretch"):
        _apply_all(dataset)
    for position, item in enumerate(recommendations):
        _render_recommendation(dataset, item, position)


def _apply_all(dataset: Dataset) -> None:
    """Apply every suggestion (re-profiling after each) and summarize what happened."""
    with st.spinner("Applying suggestions..."):
        outcome = apply_all_recommendations(dataset)
    message = f"Applied {len(outcome.applied)} suggestion(s)."
    if outcome.skipped:
        message += f" Skipped {len(outcome.skipped)}: " + "; ".join(outcome.skipped)
    run_action(lambda: None, message)


def _cached_recommendations(dataset: Dataset) -> list[Recommendation]:
    """Profile the table once per table version (keeps button clicks responsive)."""
    cache = st.session_state.get("recommendation_cache")
    if cache and cache["frame"] is dataset.current and cache["name"] == dataset.name:
        return cache["items"]
    items = recommend(dataset.current)
    st.session_state["recommendation_cache"] = {"frame": dataset.current, "name": dataset.name, "items": items}
    return items


def _render_recommendation(dataset: Dataset, item: Recommendation, position: int) -> None:
    with st.container(border=True):
        text_column, button_column = st.columns([5, 1], vertical_alignment="center")
        text_column.markdown(f"**{item.title}**  \n{_PRIORITY_BADGES[item.priority]} · {item.reason}")
        if button_column.button("Apply", key=f"rec_apply_{position}", type="primary"):
            run_action(lambda: dataset.apply(item.operation, **item.params), f"Applied: {item.title}")
        with st.expander("Code"):
            st.code(item.code(), language="python")


# ------------------------------------------------------------------------------ clean


def render_clean(dataset: Dataset) -> None:
    operation_name = st.selectbox(
        "Operation",
        list(FORMS),
        format_func=lambda name: f"{_title(name)} - {OPERATIONS[name].summary}",
        key="clean_operation",
    )
    operation = OPERATIONS[operation_name]
    # The cursor is part of the widget keys so every form resets after a step is applied.
    form_key = f"{dataset.name}|{dataset.cursor}|{operation_name}"
    params = FORMS[operation_name](dataset.current, form_key)
    if params is None:
        st.caption("Complete the options above to see the code and a preview.")
        return

    try:
        normalized = operation.normalize(dataset.current, params)
        preview = dataset.preview(operation_name, **params)
    except Exception as error:
        st.error(str(error), icon=":material/error:")
        return

    st.subheader("Code that will run")
    st.code(operation.to_code(**normalized), language="python")

    before = dataset.current
    removed = 0 if preview.removed_rows is None else len(preview.removed_rows)
    changed = 0 if preview.changed_cells is None else len(preview.changed_cells)
    metrics = st.columns(4)
    metrics[0].metric("Rows", f"{len(preview.data):,}", delta=len(preview.data) - len(before))
    metrics[1].metric("Columns", f"{preview.data.shape[1]:,}", delta=preview.data.shape[1] - before.shape[1])
    metrics[2].metric("Rows to quarantine", f"{removed:,}")
    metrics[3].metric("Cells to quarantine", f"{changed:,}")
    st.dataframe(preview.data.head(50), alt="Preview of the table after this step")

    if st.button("Apply step", type="primary", icon=":material/play_arrow:", key=f"apply_{form_key}"):
        run_action(lambda: dataset.apply(operation_name, **params), f"Applied: {_title(operation_name)}")


# ---------------------------------------------------------------------------- combine


def render_combine(workspace: Workspace) -> None:
    names = list(workspace.datasets)
    if len(names) < 2:
        st.info("Load at least two datasets (sidebar) to merge or concatenate them.")
        return
    mode = st.segmented_control("Combine by", ["Merge (join)", "Concat (stack)"], default="Merge (join)", key="combine_mode") or "Merge (join)"
    st.caption("The result is a new dataset: a snapshot of the sources' current tables.")
    if mode.startswith("Merge"):
        _render_merge(workspace, names)
    else:
        _render_concat(workspace, names)


def _render_merge(workspace: Workspace, names: list[str]) -> None:
    left_column, right_column = st.columns(2)
    left_name = left_column.selectbox("Left dataset", names, key="merge_left")
    right_name = right_column.selectbox("Right dataset", names, index=1, key="merge_right")
    left, right = workspace.datasets[left_name].current, workspace.datasets[right_name].current
    how = st.segmented_control("Join type", ["inner", "left", "right", "outer", "cross"], default="inner", key="merge_how") or "inner"
    left_on, right_on = [], []
    if how != "cross":
        left_keys = left_column.multiselect("Left key column(s)", list(left.columns), key="merge_left_on")
        right_keys = right_column.multiselect("Right key column(s)", list(right.columns), key="merge_right_on")
        if not left_keys or len(left_keys) != len(right_keys):
            st.caption("Pick the same number of key columns on each side.")
            return
        left_on, right_on = left_keys, right_keys
    suffix_columns = st.columns(2)
    suffixes = (
        suffix_columns[0].text_input("Left suffix (overlapping names)", "_x", key="merge_sx"),
        suffix_columns[1].text_input("Right suffix", "_y", key="merge_sy"),
    )
    new_name = st.text_input("New dataset name", f"{left_name}_{right_name}", key="merge_name")
    arguments = f"how={how!r}, left_on={left_on or None!r}, right_on={right_on or None!r}, suffixes={suffixes!r}"
    st.code(f"df = left_frame.merge(right_frame, {arguments})", language="python")
    if st.button("Create merged dataset", type="primary", key="merge_go"):
        run_action(
            lambda: workspace.merge(left_name, right_name, new_name, how, left_on or None, right_on or None, suffixes),
            f"Created '{new_name}'.",
        )


def _render_concat(workspace: Workspace, names: list[str]) -> None:
    chosen = st.multiselect("Datasets to combine (in order)", names, default=names[:2], key="concat_names")
    axis_label = st.segmented_control("Stack", ["rows (below each other)", "columns (side by side)"], default="rows (below each other)", key="concat_axis")
    axis = 1 if axis_label and axis_label.startswith("columns") else 0
    join = st.segmented_control("Keep", ["outer", "inner"], default="outer", key="concat_join", help="outer keeps every column/row; inner keeps only the shared ones.") or "outer"
    ignore_index = st.checkbox("Renumber the index", value=True, key="concat_ignore")
    new_name = st.text_input("New dataset name", "combined", key="concat_name")
    st.code(f"df = pd.concat([{', '.join(chosen)}], axis={axis}, join={join!r}, ignore_index={ignore_index})", language="python")
    if st.button("Create combined dataset", type="primary", key="concat_go", disabled=len(chosen) < 2):
        run_action(lambda: workspace.concat(chosen, new_name, axis, join, ignore_index), f"Created '{new_name}'.")


# ------------------------------------------------------------------------ quarantine


def render_quarantine(dataset: Dataset) -> None:
    removed, changed = dataset.removed_rows(), dataset.changed_cells()
    metrics = st.columns(2)
    metrics[0].metric("Removed rows", f"{len(removed):,}")
    metrics[1].metric("Changed cells", f"{len(changed):,}")
    st.caption("Nothing is lost: rows removed and cells changed by cleaning steps are listed here with the step and reason.")

    st.subheader("Removed rows")
    if removed.empty:
        st.write("No rows have been removed.")
    else:
        selection = st.dataframe(
            removed, on_select="rerun", selection_mode="multi-row", key=f"quarantine_rows|{dataset.name}|{dataset.cursor}",
            alt="Rows removed by cleaning steps, with the step and reason",
        )  # fmt: skip
        selected = list(selection.selection.rows)
        if st.button(f"Restore {len(selected)} selected row(s)", disabled=not selected, icon=":material/restore:"):
            run_action(lambda: dataset.restore_rows(selected), f"Restored {len(selected)} row(s).")

    st.subheader("Changed cells")
    if changed.empty:
        st.write("No cells have been changed.")
    else:
        st.dataframe(changed.astype({"before": "string", "after": "string"}), alt="Cells changed by cleaning steps, before and after")


# ---------------------------------------------------------------------------- plot


def render_plot(dataset: Dataset) -> None:
    df = dataset.current
    kind = st.selectbox("Plot type", list(PLOT_KINDS), format_func=_title, key="plot_kind")
    specs = PLOT_KINDS[kind]
    arguments: dict = {}
    _render_plot_arguments([s for s in specs if s.group == DATA], df, kind, arguments)
    style_specs = [s for s in specs if s.group == STYLE]
    if style_specs:
        with st.expander("Style options"):
            _render_plot_arguments(style_specs, df, kind, arguments)
    title = st.text_input("Title (optional)", key="plot_title")
    max_points = st.number_input(
        "Max rows to plot (larger tables are sampled; 0 = all)", min_value=0, value=DEFAULT_MAX_POINTS, step=10_000, key="plot_max_points"
    )

    try:
        figure = build_figure(df, kind, arguments, title or None, max_points or None)
    except (ValueError, KeyError, TypeError) as error:
        st.info(f"{error}  (* = required)")
        return
    st.plotly_chart(figure, key=f"plot_chart|{kind}", alt=f"{_title(kind)} plot of {dataset.name}")
    code = plot_code(kind, arguments, title or None, max_points or None)
    with st.expander("Code", expanded=True):
        st.code(code, language="python")
    st.download_button("Download chart (HTML)", figure.to_html(), file_name=f"{kind}.html", mime="text/html", icon=":material/download:")


def _render_plot_arguments(specs: list[PlotArgument], df: pd.DataFrame, kind: str, arguments: dict) -> None:
    """Draw one widget per argument, matched to its type, and collect values into ``arguments``."""
    grid = st.columns(3)
    for position, spec in enumerate(specs):
        with grid[position % 3]:
            arguments.update(_render_plot_argument(spec, df, kind))


def _render_plot_argument(spec: PlotArgument, df: pd.DataFrame, kind: str) -> dict:
    """Widget(s) for one argument. Returns {argument name: value} entries to pass on."""
    label = spec.name + (" *" if spec.required else "")
    key = f"plot|{kind}|{spec.name}"
    help_text = spec.help or None
    blank = lambda option: "-" if option is None else option  # noqa: E731 - tiny display helper

    if spec.kind == "column":
        column = st.selectbox(label, [None, *df.columns], key=key, help=help_text, format_func=blank)
        if not spec.fixed:
            return {spec.name: column}
        fixed = _render_fixed_value(spec, key, disabled=column is not None)
        return {spec.name: column, fixed_key(spec.name): fixed}
    if spec.kind == "columns":
        return {spec.name: st.multiselect(label, list(df.columns), key=key, help=help_text)}
    if spec.kind == "number":
        is_float = spec.number_type == "float"
        number_arguments = {
            "min_value": float(spec.minimum) if is_float and spec.minimum is not None else spec.minimum,
            "max_value": float(spec.maximum) if is_float and spec.maximum is not None else spec.maximum,
            "step": float(spec.step) if is_float and spec.step else spec.step or (None if is_float else 1),
        }
        return {spec.name: st.number_input(label, value=None, key=key, help=help_text, placeholder="auto", **number_arguments)}
    if spec.kind == "boolean":
        return {spec.name: st.checkbox(label, key=key, help=help_text)}
    return {spec.name: st.selectbox(label, [None, *spec.choices], key=key, help=help_text, format_func=blank)}


def _render_fixed_value(spec: PlotArgument, key: str, disabled: bool):
    """Input for a single value applied to every point (instead of mapping a column)."""
    label = f"or fixed {spec.name}"
    note = "Pick a column above, or clear it to use a fixed value." if disabled else None
    if spec.fixed == "color":
        return st.text_input(label, key=f"{key}_fixed", disabled=disabled, placeholder="red or #1f77b4", help=note or "Any CSS color name or hex code.")
    if spec.fixed == "size":
        return st.number_input(label, min_value=1, value=None, step=1, key=f"{key}_fixed", disabled=disabled, placeholder="e.g. 12", help=note)
    return st.selectbox(label, [None, *MARKER_SYMBOLS], key=f"{key}_fixed", disabled=disabled, format_func=lambda option: "-" if option is None else option, help=note)


# ---------------------------------------------------------------------------- code


def render_code(dataset: Dataset) -> None:
    st.subheader("Full script")
    st.caption("Rebuilds the current table from the source data. Quarantine tracking is not part of the script.")
    script = dataset.export_script()
    st.code(script, language="python")
    download_columns = st.columns(2)
    download_columns[0].download_button("Download script (.py)", script, file_name=f"{dataset.name}_clean.py", icon=":material/download:")
    download_columns[1].download_button("Download steps (.json)", dataset.steps_to_json(), file_name=f"{dataset.name}_steps.json", icon=":material/download:")

    uploaded = st.file_uploader("Replay saved steps (.json) on this dataset", type=["json"], key=f"steps_upload|{dataset.name}")
    if uploaded is not None and st.button("Replace history with these steps", key="steps_replay"):
        run_action(lambda: dataset.load_steps_json(uploaded.getvalue().decode("utf-8")), "Steps replayed.")

    st.subheader(f"History ({len(dataset.steps)} step(s))")
    if not dataset.steps:
        st.write("No steps yet. Use Clean, Recommendations or the editable table.")
    for number, step in enumerate(dataset.steps, start=1):
        with st.expander(f"{number}. {step.label(70)}"):
            st.code(dataset.step_code(number), language="python")
            edited = st.text_area(
                "Parameters (JSON) - edit and save to change this step; later steps are replayed",
                json.dumps(step.params, indent=2, default=str),
                key=f"step_params|{dataset.name}|{number}|{dataset.cursor}",
            )
            save_column, remove_column, _ = st.columns([1, 1, 4])
            if save_column.button("Save changes", key=f"step_save_{number}"):
                run_action(lambda: dataset.edit_step(number - 1, json.loads(edited)), f"Step {number} updated.")
            if remove_column.button("Remove step", key=f"step_remove_{number}", icon=":material/delete:"):
                run_action(lambda: dataset.remove_step(number - 1), f"Step {number} removed.")
