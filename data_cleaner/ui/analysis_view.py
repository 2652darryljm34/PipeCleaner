"""The Analyze section: look at the data without changing it.

Everything here is read-only. Results can be downloaded, plotted, or saved as a *new*
dataset; the dataset being analyzed is never modified.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from data_cleaner.core import analysis
from data_cleaner.core.analysis import AnalysisResult
from data_cleaner.core.dataset import Dataset
from data_cleaner.core.operations.reshape import AGGREGATION_FUNCTIONS
from data_cleaner.core.workspace import Workspace

from .state import run_action
from .views import render_plotter

VIEWS = ["Overview", "Statistics", "Value counts", "Group by", "Pivot", "Correlation"]
DEFAULT_VALUE_COUNT_LIMIT = 25


def render_analyze(workspace: Workspace, dataset: Dataset) -> None:
    st.caption("Analysis never changes your data. Save a result as a new dataset if you want to keep working with it.")
    view = st.segmented_control("Analysis", VIEWS, default=VIEWS[0], key="analysis_view", label_visibility="collapsed") or VIEWS[0]
    renderers: dict[str, Callable[[Workspace, Dataset], None]] = {
        "Overview": _render_overview,
        "Statistics": _render_statistics,
        "Value counts": _render_value_counts,
        "Group by": _render_group_by,
        "Pivot": _render_pivot,
        "Correlation": _render_correlation,
    }
    renderers[view](workspace, dataset)


# ------------------------------------------------------------------------- shared


def _show_result(
    result: AnalysisResult, name: str, key: str, *, workspace: Workspace | None = None, dataset: Dataset | None = None
) -> None:
    """Table, code, CSV download, optional chart and (for group/pivot) save-as-dataset."""
    st.dataframe(result.table, alt=f"{name} analysis result")
    with st.expander("Code"):
        st.code(result.code, language="python")
    action_columns = st.columns(2)
    action_columns[0].download_button(
        "Download CSV", result.table.to_csv(index=not isinstance(result.table.index, pd.RangeIndex)),
        file_name=f"{name}.csv", mime="text/csv", icon=":material/download:", key=f"{key}_download",
    )  # fmt: skip
    if workspace is not None and dataset is not None:
        with st.container(border=True):
            new_name = st.text_input("Save as a new dataset", value=f"{dataset.name}_{name}", key=f"{key}_save_name")
            if st.button("Save dataset", icon=":material/save:", key=f"{key}_save", disabled=not new_name.strip()):
                run_action(
                    lambda: workspace.save_analysis(dataset.name, new_name.strip(), result),
                    f"Saved '{new_name.strip()}' as a new dataset. Your original dataset is unchanged.",
                )
    if st.toggle("Plot this result", key=f"{key}_plot"):
        render_plotter(result.table.reset_index() if result.table.index.name else result.table, name, key_prefix=f"{key}_plotter")


def _pick(label: str, options: list[str], key: str, default: list[str] | None = None) -> list[str]:
    return st.multiselect(label, options, default=default, key=key)


# -------------------------------------------------------------------------- views


def _render_overview(workspace: Workspace, dataset: Dataset) -> None:
    df = dataset.current
    columns = _pick("Columns (empty = all)", list(df.columns), f"overview|{dataset.name}")
    _show_result(analysis.column_overview(df, columns or None), "overview", "overview")


def _render_statistics(workspace: Workspace, dataset: Dataset) -> None:
    df = dataset.current
    numeric = analysis.numeric_columns(df)
    if not numeric:
        st.info("This table has no numeric columns.")
        return
    columns = _pick("Numeric columns (empty = all)", numeric, f"stats|{dataset.name}")
    st.caption("count, sum, mean, median, standard deviation (std), variance (var), min, quartiles (q1, q3), max.")
    _show_result(analysis.numeric_statistics(df, columns or None), "statistics", "statistics")


def _render_value_counts(workspace: Workspace, dataset: Dataset) -> None:
    df = dataset.current
    column = st.selectbox("Column", list(df.columns), key=f"counts_column|{dataset.name}")
    options = st.columns(2)
    include_nulls = options[0].checkbox("Count nulls too", key="counts_nulls")
    limit = options[1].number_input("Show top (0 = all)", min_value=0, value=DEFAULT_VALUE_COUNT_LIMIT, step=5, key="counts_limit")
    result = analysis.value_counts(df, column, include_nulls, int(limit) or None)
    st.metric("Distinct values", f"{df[column].nunique(dropna=not include_nulls):,}")
    _show_result(result, f"{column}_counts", "counts")
    st.plotly_chart(
        px.bar(result.table, x="value", y="count", title=f"Most common values of {column}"),
        key="counts_chart", alt=f"Counts of each distinct value in {column}",
    )  # fmt: skip


def _render_group_by(workspace: Workspace, dataset: Dataset) -> None:
    df = dataset.current
    by = _pick("Group by", list(df.columns), f"group_by|{dataset.name}")
    remaining = [column for column in df.columns if column not in by]
    numeric_default = [column for column in analysis.numeric_columns(df) if column in remaining][:1]
    values = _pick("Summarize these columns", remaining, f"group_values|{dataset.name}", default=numeric_default)
    functions = _pick("With these functions", AGGREGATION_FUNCTIONS, "group_functions", default=["sum", "mean"])
    include_row_count = st.checkbox("Include the number of rows per group", value=True, key="group_row_count")
    if not by or not ((values and functions) or include_row_count):
        st.caption("Choose at least one column to group by, and something to summarize or count.")
        return
    try:
        result = analysis.grouped_summary(df, by, {column: functions for column in values}, include_row_count)
    except Exception as error:
        st.error(str(error), icon=":material/error:")
        return
    st.metric("Groups", f"{len(result.table):,}")
    _show_result(result, "group_summary", "group", workspace=workspace, dataset=dataset)


def _render_pivot(workspace: Workspace, dataset: Dataset) -> None:
    df = dataset.current
    index = _pick("Rows", list(df.columns), f"pivot_index|{dataset.name}")
    columns = _pick("Columns (their values become headers)", list(df.columns), f"pivot_columns|{dataset.name}")
    values = _pick("Values to summarize", list(df.columns), f"pivot_values|{dataset.name}")
    aggfunc = st.selectbox("Summarize with", AGGREGATION_FUNCTIONS, key="pivot_aggfunc")
    if not (index and columns and values):
        st.caption("Choose rows, columns and values.")
        return
    try:
        result = analysis.pivot_summary(df, index, columns, values, aggfunc)
    except Exception as error:
        st.error(str(error), icon=":material/error:")
        return
    _show_result(result, "pivot", "pivot", workspace=workspace, dataset=dataset)


def _render_correlation(workspace: Workspace, dataset: Dataset) -> None:
    df = dataset.current
    if len(analysis.numeric_columns(df)) < 2:
        st.info("Correlation needs at least two numeric columns.")
        return
    method = st.segmented_control("Method", ["pearson", "spearman", "kendall"], default="pearson", key="correlation_method") or "pearson"
    result = analysis.correlations(df, method)
    st.plotly_chart(
        px.imshow(result.table, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1),
        key="correlation_chart", alt="Correlation between each pair of numeric columns",
    )  # fmt: skip
    st.dataframe(result.table, alt="Correlation matrix")
    with st.expander("Code"):
        st.code(result.code, language="python")
