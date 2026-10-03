"""One Streamlit input form per cleaning operation.

Each ``form_*`` function renders widgets for ``df`` and returns the operation's params,
or ``None`` while the inputs are incomplete. The clean view shows the generated code and
an Apply button once params are available, so forms never apply anything themselves.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from data_cleaner.core.operations import parse_scalar
from data_cleaner.core.operations.cleaning import (
    DTYPE_CHOICES,
    FILL_STRATEGIES,
    FILTER_OPERATORS,
    NULL_OPERATORS,
)

FormRenderer = Callable[[pd.DataFrame, str], "dict | None"]
MAX_FILTER_CONDITIONS = 8


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def _text_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])
    ]


def _describe_column(df: pd.DataFrame, column: str) -> str:
    """Label like 'age (int64, 3 nulls)' so users see types and gaps while choosing."""
    nulls = int(df[column].isna().sum())
    return f"{column} ({df[column].dtype}{f', {nulls} nulls' if nulls else ''})"


def _pick_columns(label: str, df: pd.DataFrame, key: str, default: list[str] | None = None, help: str | None = None) -> list[str]:
    return st.multiselect(
        label, list(df.columns), default=default, key=key, help=help,
        format_func=lambda column: _describe_column(df, column),
    )  # fmt: skip


# ---------------------------------------------------------------------------- forms


def form_rename_columns(df: pd.DataFrame, key: str) -> dict | None:
    names = pd.DataFrame({"column": list(df.columns), "new_name": list(df.columns)})
    edited = st.data_editor(
        names, disabled=["column"], hide_index=True, key=f"{key}_names",
        alt="Column names; edit the new_name cells to rename columns",
    )  # fmt: skip
    mapping = {old: new for old, new in zip(edited["column"], edited["new_name"]) if new and new != old}
    return {"mapping": mapping} if mapping else None


def form_drop_duplicates(df: pd.DataFrame, key: str) -> dict:
    subset = _pick_columns("Compare these columns (empty = all columns)", df, f"{key}_subset")
    keep = st.segmented_control(
        "Keep", ["first", "last", "none"], default="first", key=f"{key}_keep",
        help="'none' drops every copy of a duplicated row.",
    )  # fmt: skip
    return {"subset": subset or None, "keep": keep or "first"}


def form_drop_nulls(df: pd.DataFrame, key: str) -> dict:
    subset = _pick_columns("Check these columns (empty = all columns)", df, f"{key}_subset")
    how = st.segmented_control(
        "Drop a row when", ["any", "all"], default="any", key=f"{key}_how",
        help="'any' = at least one checked column is null; 'all' = every checked column is null.",
    )  # fmt: skip
    return {"subset": subset or None, "how": how or "any"}


def form_filter_rows(df: pd.DataFrame, key: str) -> dict | None:
    st.caption("Rows matching the conditions are kept; the rest go to quarantine.")
    count = st.number_input("Number of conditions", 1, MAX_FILTER_CONDITIONS, 1, key=f"{key}_count")
    conditions = []
    for position in range(int(count)):
        column_input, operator_input, value_input = st.columns([2, 2, 3])
        column = column_input.selectbox("Column", list(df.columns), key=f"{key}_col{position}")
        operator = operator_input.selectbox("Operator", FILTER_OPERATORS, key=f"{key}_op{position}")
        needs_value = operator not in NULL_OPERATORS
        value = value_input.text_input(
            "Value", key=f"{key}_val{position}", disabled=not needs_value,
            help="For 'is in', separate values with commas.",
        )  # fmt: skip
        if needs_value and value == "":
            return None  # incomplete
        conditions.append({"column": column, "operator": operator, **({"value": value} if needs_value else {})})
    combine = "and"
    if count > 1:
        combine = st.segmented_control("Combine conditions with", ["and", "or"], default="and", key=f"{key}_combine") or "and"
    return {"conditions": conditions, "combine": combine}


def form_convert_dtype(df: pd.DataFrame, key: str) -> dict:
    column = st.selectbox("Column", list(df.columns), key=f"{key}_col", format_func=lambda c: _describe_column(df, c))
    dtype = st.selectbox("Convert to", DTYPE_CHOICES, key=f"{key}_dtype")
    params = {"column": column, "dtype": dtype}
    if dtype == "datetime":
        datetime_format = st.text_input("Date format (optional)", placeholder="%d/%m/%Y", key=f"{key}_format")
        if datetime_format:
            params["datetime_format"] = datetime_format
    st.caption("Values that cannot be converted become null and are listed in quarantine.")
    return params


def form_fill_nulls(df: pd.DataFrame, key: str) -> dict | None:
    with_nulls = [c for c in df.columns if df[c].isna().any()]
    if not with_nulls:
        st.info("This table has no null values.")
        return None
    columns = _pick_columns("Columns to fill", df, f"{key}_cols", default=with_nulls[:1])
    strategy = st.selectbox("Fill with", FILL_STRATEGIES, key=f"{key}_strategy")
    params = {"columns": columns, "strategy": strategy}
    if strategy == "value":
        value = st.text_input("Value", key=f"{key}_value")
        if value == "":
            return None
        params["value"] = value
    return params if columns else None


def form_replace_values(df: pd.DataFrame, key: str) -> dict | None:
    columns = _pick_columns("Search these columns (empty = all columns)", df, f"{key}_cols")
    to_replace = st.text_input("Replace these values", key=f"{key}_old", help="Separate several values with commas.")
    make_null = st.checkbox("Replace with null (blank)", key=f"{key}_null")
    value = None if make_null else st.text_input("Replacement value", key=f"{key}_new")
    if not to_replace or (not make_null and value == ""):
        return None
    return {"columns": columns or None, "to_replace": to_replace, "value": value}


def form_clean_text(df: pd.DataFrame, key: str) -> dict | None:
    text_columns = _text_columns(df)
    columns = st.multiselect("Text columns", text_columns, default=text_columns[:1], key=f"{key}_cols")
    strip = st.checkbox("Trim leading/trailing whitespace", value=True, key=f"{key}_strip")
    collapse = st.checkbox("Collapse repeated spaces", key=f"{key}_collapse")
    case = st.segmented_control("Letter case", ["none", "lower", "upper", "title"], default="none", key=f"{key}_case")
    return {"columns": columns, "strip": strip, "collapse_spaces": collapse, "case": case or "none"} if columns else None


def form_create_bins(df: pd.DataFrame, key: str) -> dict | None:
    numeric = _numeric_columns(df)
    if not numeric:
        st.info("Binning needs a numeric column.")
        return None
    column = st.selectbox("Numeric column", numeric, key=f"{key}_col")
    method = st.segmented_control(
        "Method", ["equal_width", "quantile", "custom"], default="equal_width", key=f"{key}_method",
        help="equal_width: same-size ranges. quantile: same number of rows per bin. custom: your own edges.",
    ) or "equal_width"  # fmt: skip
    if method == "custom":
        bins = st.text_input("Bin edges (comma-separated)", placeholder="0, 18, 65, 120", key=f"{key}_edges")
        if not bins.strip():
            return None
    else:
        bins = st.number_input("Number of bins", 2, 100, 5, key=f"{key}_bins")
    labels = st.text_input("Labels (optional, comma-separated)", key=f"{key}_labels")
    new_column = st.text_input("New column name", value=f"{column}_binned", key=f"{key}_new_{column}")
    right = st.checkbox("Bins include their right edge", value=True, key=f"{key}_right")
    return {"column": column, "method": method, "bins": bins, "labels": labels or None, "new_column": new_column, "right": right}


def form_add_column(df: pd.DataFrame, key: str) -> dict | None:
    name = st.text_input("New column name", key=f"{key}_name")
    mode = st.segmented_control("Fill from", ["expression", "constant"], default="expression", key=f"{key}_mode") or "expression"
    if mode == "expression":
        expression = st.text_input(
            "Expression", placeholder="price * quantity", key=f"{key}_expr",
            help="Uses pandas eval: column names, + - * / **, comparisons, and/or. Wrap odd names in `backticks`.",
        )  # fmt: skip
        params = {"name": name, "mode": mode, "expression": expression}
        return params if name and expression else None
    value = st.text_input("Constant value", key=f"{key}_value")
    return {"name": name, "mode": mode, "value": parse_scalar(value)} if name else None


def form_drop_columns(df: pd.DataFrame, key: str) -> dict | None:
    columns = _pick_columns("Columns to remove", df, f"{key}_cols")
    return {"columns": columns} if columns else None


def form_sort_values(df: pd.DataFrame, key: str) -> dict | None:
    by = _pick_columns("Sort by (in order)", df, f"{key}_by")
    ascending = st.toggle("Ascending", value=True, key=f"{key}_asc")
    return {"by": by, "ascending": ascending} if by else None


def form_stack(df: pd.DataFrame, key: str) -> dict | None:
    id_columns = _pick_columns("Identifier columns (kept as-is)", df, f"{key}_ids", help="Every other column is stacked into rows.")
    var_name = st.text_input("Name for old column names", value="variable", key=f"{key}_var")
    value_name = st.text_input("Name for values", value="value", key=f"{key}_val")
    return {"id_columns": id_columns, "var_name": var_name, "value_name": value_name} if id_columns else None


def form_melt(df: pd.DataFrame, key: str) -> dict | None:
    id_vars = _pick_columns("Identifier columns (kept as-is)", df, f"{key}_ids")
    value_vars = _pick_columns("Columns to unpivot (empty = all others)", df, f"{key}_vals")
    var_name = st.text_input("Name for old column names", value="variable", key=f"{key}_var")
    value_name = st.text_input("Name for values", value="value", key=f"{key}_valname")
    if len(id_vars) == len(df.columns):
        return None
    return {"id_vars": id_vars or None, "value_vars": value_vars or None, "var_name": var_name, "value_name": value_name}


# Operations offered in the Clean section. Manual edits (set_cells, drop_rows, add_rows,
# restore_rows) come from the editable table and the quarantine view instead; group by and
# pivot are analysis (see ui/analysis_view.py) because they would replace the table.
FORMS: dict[str, FormRenderer] = {
    "rename_columns": form_rename_columns,
    "drop_duplicates": form_drop_duplicates,
    "drop_nulls": form_drop_nulls,
    "filter_rows": form_filter_rows,
    "convert_dtype": form_convert_dtype,
    "fill_nulls": form_fill_nulls,
    "replace_values": form_replace_values,
    "clean_text": form_clean_text,
    "create_bins": form_create_bins,
    "add_column": form_add_column,
    "drop_columns": form_drop_columns,
    "sort_values": form_sort_values,
    "stack": form_stack,
    "melt": form_melt,
}
