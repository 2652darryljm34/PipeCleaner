"""Operations that clean rows and values: duplicates, nulls, filters, dtypes, text."""

from __future__ import annotations

import functools
import operator
from typing import Any

import numpy as np
import pandas as pd

from .base import (
    Operation,
    OperationError,
    OperationResult,
    Params,
    changed_cells_frame,
    code_literal,
    concat_changed_cells,
    removed_rows_frame,
    require_columns,
)


class RenameColumns(Operation):
    name = "rename_columns"
    summary = "Rename one or more columns."
    parameter_docs = {"mapping": "dict of {old_name: new_name}"}

    def normalize(self, df: pd.DataFrame, params: Params) -> Params:
        mapping = {old: new for old, new in params["mapping"].items() if new and old != new}
        return {"mapping": mapping}

    def apply(self, df: pd.DataFrame, mapping: dict[str, str]) -> OperationResult:
        require_columns(df, list(mapping))
        renamed = df.rename(columns=mapping)
        if renamed.columns.duplicated().any():
            raise OperationError("Renaming would create duplicate column names.")
        return OperationResult(renamed)

    def to_code(self, mapping: dict[str, str]) -> str:
        return f"df = df.rename(columns={code_literal(mapping)})"


class DropDuplicates(Operation):
    name = "drop_duplicates"
    summary = "Drop duplicate rows (removed rows go to quarantine)."
    parameter_docs = {
        "subset": "columns that define a duplicate (default: all columns)",
        "keep": "'first', 'last' or 'none' (drop every copy)",
    }
    list_parameters = ("subset",)

    _KEEP_ARGUMENT = {"first": "first", "last": "last", "none": False}

    def apply(
        self, df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first"
    ) -> OperationResult:
        require_columns(df, subset)
        if keep not in self._KEEP_ARGUMENT:
            raise OperationError(f"keep must be one of {list(self._KEEP_ARGUMENT)}")
        is_duplicate = df.duplicated(subset=subset or None, keep=self._KEEP_ARGUMENT[keep])
        scope = f" on {subset}" if subset else ""
        removed = removed_rows_frame(df, is_duplicate, f"duplicate row{scope} (keep={keep})")
        return OperationResult(df.loc[~is_duplicate.to_numpy()], removed_rows=removed)

    def to_code(self, subset: list[str] | None = None, keep: str = "first") -> str:
        subset_argument = f"subset={code_literal(subset)}, " if subset else ""
        return f"df = df.drop_duplicates({subset_argument}keep={code_literal(self._KEEP_ARGUMENT[keep])})"


class DropNulls(Operation):
    name = "drop_nulls"
    summary = "Drop rows containing nulls (removed rows go to quarantine)."
    parameter_docs = {
        "subset": "columns to check (default: all columns)",
        "how": "'any' drops rows with at least one null, 'all' only fully-null rows",
    }
    list_parameters = ("subset",)

    def apply(
        self, df: pd.DataFrame, subset: list[str] | None = None, how: str = "any"
    ) -> OperationResult:
        require_columns(df, subset)
        if how not in {"any", "all"}:
            raise OperationError("how must be 'any' or 'all'")
        columns = subset or list(df.columns)
        null_flags = df[columns].isna()
        has_null = null_flags.any(axis=1) if how == "any" else null_flags.all(axis=1)

        # Name the offending columns for each quarantined row (only a few rows, so a loop is fine).
        removed_flags = null_flags.loc[has_null.to_numpy()].to_numpy()
        reasons = [
            "null in: " + ", ".join(col for col, is_null in zip(columns, row) if is_null)
            for row in removed_flags
        ]
        removed = removed_rows_frame(df, has_null, reasons)
        return OperationResult(df.loc[~has_null.to_numpy()], removed_rows=removed)

    def to_code(self, subset: list[str] | None = None, how: str = "any") -> str:
        subset_argument = f"subset={code_literal(subset)}, " if subset else ""
        return f"df = df.dropna({subset_argument}how={code_literal(how)})"


# ------------------------------------------------------------------ filtering rows

NULL_OPERATORS = {"is null", "is not null"}
FILTER_OPERATORS = [
    "==", "!=", ">", ">=", "<", "<=",
    "contains", "does not contain", "starts with", "ends with",
    "is in", "is null", "is not null",
]  # fmt: skip
_COMPARISONS = {
    "==": operator.eq, "!=": operator.ne,
    ">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
}  # fmt: skip


def _coerce_filter_value(series: pd.Series, value: Any) -> Any:
    """Convert a user-typed filter value to something comparable with ``series``."""
    if isinstance(value, (list, tuple)):
        return [_coerce_filter_value(series, item) for item in value]
    if not isinstance(value, str):
        return value
    if pd.api.types.is_bool_dtype(series):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    if pd.api.types.is_numeric_dtype(series):
        number = pd.to_numeric(value, errors="coerce")
        if pd.isna(number):
            raise OperationError(f"'{value}' is not a number (column '{series.name}').")
        return number.item()
    return value  # text and datetimes compare fine as strings (pandas parses dates)


def _condition_mask(df: pd.DataFrame, condition: dict[str, Any]) -> pd.Series:
    """Boolean mask for one filter condition; nulls never match unless asked for."""
    column = df[condition["column"]]
    operator_name, value = condition["operator"], condition.get("value")
    as_text = column.astype("string")
    if operator_name in _COMPARISONS:
        mask = _COMPARISONS[operator_name](column, value)
    elif operator_name == "contains":
        mask = as_text.str.contains(str(value), regex=False, na=False)
    elif operator_name == "does not contain":
        mask = ~as_text.str.contains(str(value), regex=False, na=False)
    elif operator_name == "starts with":
        mask = as_text.str.startswith(str(value), na=False)
    elif operator_name == "ends with":
        mask = as_text.str.endswith(str(value), na=False)
    elif operator_name == "is in":
        mask = column.isin(value)
    elif operator_name == "is null":
        mask = column.isna()
    elif operator_name == "is not null":
        mask = column.notna()
    else:
        raise OperationError(f"Unknown operator '{operator_name}'. Use one of {FILTER_OPERATORS}")
    return mask.fillna(False).astype(bool)


def _condition_code(condition: dict[str, Any]) -> str:
    """pandas expression equivalent to ``_condition_mask``."""
    column = f"df[{code_literal(condition['column'])}]"
    operator_name, value = condition["operator"], code_literal(condition.get("value"))
    text = f"{column}.astype('string')"
    expressions = {
        "contains": f"{text}.str.contains({value}, regex=False, na=False)",
        "does not contain": f"~{text}.str.contains({value}, regex=False, na=False)",
        "starts with": f"{text}.str.startswith({value}, na=False)",
        "ends with": f"{text}.str.endswith({value}, na=False)",
        "is in": f"{column}.isin({value})",
        "is null": f"{column}.isna()",
        "is not null": f"{column}.notna()",
    }
    if operator_name in _COMPARISONS:
        return f"({column} {operator_name} {value})"
    return f"({expressions[operator_name]})"


def describe_condition(condition: dict[str, Any]) -> str:
    value = "" if condition["operator"] in NULL_OPERATORS else f" {condition.get('value')!r}"
    return f"{condition['column']} {condition['operator']}{value}"


class FilterRows(Operation):
    name = "filter_rows"
    summary = "Keep only rows matching conditions (the rest go to quarantine)."
    parameter_docs = {
        "conditions": "list of {column, operator, value}; operators: " + ", ".join(FILTER_OPERATORS),
        "combine": "'and' (all must match) or 'or' (any may match)",
    }

    def normalize(self, df: pd.DataFrame, params: Params) -> Params:
        conditions = []
        for condition in params["conditions"]:
            require_columns(df, [condition["column"]])
            clean = {"column": condition["column"], "operator": condition["operator"]}
            if condition["operator"] not in NULL_OPERATORS:
                value = condition.get("value")
                if condition["operator"] == "is in" and isinstance(value, str):
                    value = [part.strip() for part in value.split(",")]
                if condition["operator"] in {"contains", "does not contain", "starts with", "ends with"}:
                    clean["value"] = str(value)
                else:
                    clean["value"] = _coerce_filter_value(df[condition["column"]], value)
            conditions.append(clean)
        return {"conditions": conditions, "combine": params.get("combine", "and")}

    def apply(
        self, df: pd.DataFrame, conditions: list[dict[str, Any]], combine: str = "and"
    ) -> OperationResult:
        if not conditions:
            raise OperationError("Add at least one condition.")
        combiner = {"and": operator.and_, "or": operator.or_}.get(combine)
        if combiner is None:
            raise OperationError("combine must be 'and' or 'or'")
        require_columns(df, [condition["column"] for condition in conditions])
        keep = functools.reduce(combiner, (_condition_mask(df, c) for c in conditions))
        summary = f" {combine} ".join(describe_condition(c) for c in conditions)
        removed = removed_rows_frame(df, ~keep, f"did not match: {summary}")
        return OperationResult(df.loc[keep.to_numpy()], removed_rows=removed)

    def to_code(self, conditions: list[dict[str, Any]], combine: str = "and") -> str:
        joiner = " & " if combine == "and" else " | "
        mask = joiner.join(_condition_code(c) for c in conditions)
        return f"keep_row = {mask}\ndf = df[keep_row.fillna(False).astype(bool)]"


# ------------------------------------------------------------------- dtype changes

DTYPE_CHOICES = ["int", "float", "string", "category", "bool", "datetime"]
_BOOLEAN_WORDS = {
    "true": True, "t": True, "yes": True, "y": True, "1": True,
    "false": False, "f": False, "no": False, "n": False, "0": False,
}  # fmt: skip


class ConvertDType(Operation):
    name = "convert_dtype"
    summary = "Convert a column's data type (values that cannot convert become null)."
    parameter_docs = {
        "column": "column to convert",
        "dtype": f"one of {DTYPE_CHOICES}",
        "datetime_format": "optional strptime format when dtype is datetime",
    }

    def apply(
        self, df: pd.DataFrame, column: str, dtype: str, datetime_format: str | None = None
    ) -> OperationResult:
        require_columns(df, [column])
        if dtype not in DTYPE_CHOICES:
            raise OperationError(f"dtype must be one of {DTYPE_CHOICES}")
        before = df[column]
        if dtype == "int":
            numeric = pd.to_numeric(before, errors="coerce")
            converted = numeric.where(numeric % 1 == 0).astype("Int64")
        elif dtype == "float":
            converted = pd.to_numeric(before, errors="coerce").astype("float64")
        elif dtype == "string":
            converted = before.astype("string")
        elif dtype == "category":
            converted = before.astype("category")
        elif dtype == "bool":
            lowered = before.astype("string").str.strip().str.lower()
            converted = lowered.map(_BOOLEAN_WORDS).astype("boolean")
        else:  # datetime
            converted = pd.to_datetime(before, errors="coerce", format=datetime_format or None)

        # A value that existed before but is null now could not be converted: quarantine it.
        failed = before.notna().to_numpy() & converted.isna().to_numpy()
        failed_cells = changed_cells_frame(
            before[failed], converted[failed], column, f"could not convert to {dtype}"
        )
        result = df.copy()
        result[column] = converted
        return OperationResult(result, changed_cells=failed_cells if not failed_cells.empty else None)

    def to_code(self, column: str, dtype: str, datetime_format: str | None = None) -> str:
        target = f"df[{code_literal(column)}]"
        if dtype == "int":
            return (
                f"numeric_values = pd.to_numeric({target}, errors='coerce')\n"
                f"{target} = numeric_values.where(numeric_values % 1 == 0).astype('Int64')"
            )
        if dtype == "float":
            return f"{target} = pd.to_numeric({target}, errors='coerce').astype('float64')"
        if dtype == "string":
            return f"{target} = {target}.astype('string')"
        if dtype == "category":
            return f"{target} = {target}.astype('category')"
        if dtype == "bool":
            return (
                f"boolean_words = {code_literal(_BOOLEAN_WORDS)}\n"
                f"{target} = {target}.astype('string').str.strip().str.lower()"
                f".map(boolean_words).astype('boolean')"
            )
        format_argument = f", format={code_literal(datetime_format)}" if datetime_format else ""
        return f"{target} = pd.to_datetime({target}, errors='coerce'{format_argument})"


# --------------------------------------------------------------- filling / replacing

FILL_STRATEGIES = ["value", "mean", "median", "mode", "ffill", "bfill"]


class FillNulls(Operation):
    name = "fill_nulls"
    summary = "Fill nulls in columns (changed cells go to quarantine)."
    parameter_docs = {
        "columns": "columns to fill",
        "strategy": f"one of {FILL_STRATEGIES}",
        "value": "fill value when strategy is 'value' (typed text is coerced per column)",
    }
    list_parameters = ("columns",)

    def normalize(self, df: pd.DataFrame, params: Params) -> Params:
        normalized = super().normalize(df, params)
        if normalized.get("strategy") == "value" and "value" in normalized:  # idempotent
            require_columns(df, normalized["columns"])
            raw_value = normalized.pop("value")
            normalized["fill_values"] = {
                column: _coerce_filter_value(df[column], raw_value) for column in normalized["columns"]
            }
        return normalized

    def apply(
        self,
        df: pd.DataFrame,
        columns: list[str],
        strategy: str,
        fill_values: dict[str, Any] | None = None,
    ) -> OperationResult:
        require_columns(df, columns)
        if strategy not in FILL_STRATEGIES:
            raise OperationError(f"strategy must be one of {FILL_STRATEGIES}")
        result, changes = df.copy(), []
        for column in columns:
            before = df[column]
            if strategy == "value":
                after = before.fillna((fill_values or {})[column])
            elif strategy in {"mean", "median"}:
                if not pd.api.types.is_numeric_dtype(before):
                    raise OperationError(f"Cannot take the {strategy} of non-numeric '{column}'.")
                after = before.fillna(getattr(before, strategy)())
            elif strategy == "mode":
                if before.dropna().empty:
                    raise OperationError(f"Column '{column}' is entirely null; it has no mode.")
                after = before.fillna(before.mode().iloc[0])
            else:
                after = getattr(before, strategy)()
            result[column] = after
            changes.append(changed_cells_frame(before, after, column, f"null filled ({strategy})"))
        return OperationResult(result, changed_cells=concat_changed_cells(changes))

    def to_code(
        self, columns: list[str], strategy: str, fill_values: dict[str, Any] | None = None
    ) -> str:
        lines = []
        for column in columns:
            target = f"df[{code_literal(column)}]"
            if strategy == "value":
                fill = f"{target}.fillna({code_literal((fill_values or {})[column])})"
            elif strategy in {"mean", "median"}:
                fill = f"{target}.fillna({target}.{strategy}())"
            elif strategy == "mode":
                fill = f"{target}.fillna({target}.mode().iloc[0])"
            else:
                fill = f"{target}.{strategy}()"
            lines.append(f"{target} = {fill}")
        return "\n".join(lines)


class ReplaceValues(Operation):
    name = "replace_values"
    summary = "Replace specific values (or turn them into nulls)."
    parameter_docs = {
        "columns": "columns to search (default: all columns)",
        "to_replace": "value or list of values to replace",
        "value": "replacement; use null/None to blank them out",
    }
    list_parameters = ("columns", "to_replace")

    def normalize(self, df: pd.DataFrame, params: Params) -> Params:
        normalized = super().normalize(df, params)
        columns = normalized.get("columns") or list(df.columns)
        require_columns(df, columns)
        # Typed text like "99" should match numbers when every searched column is numeric.
        if all(pd.api.types.is_numeric_dtype(df[column]) for column in columns):
            normalized["to_replace"] = [_coerce_filter_value(df[columns[0]], v) for v in normalized["to_replace"]]
            if isinstance(normalized.get("value"), str):
                normalized["value"] = _coerce_filter_value(df[columns[0]], normalized["value"])
        return normalized

    def apply(
        self,
        df: pd.DataFrame,
        to_replace: list[Any],
        value: Any = None,
        columns: list[str] | None = None,
    ) -> OperationResult:
        columns = columns or list(df.columns)
        require_columns(df, columns)
        replacement = np.nan if value is None else value
        result, changes = df.copy(), []
        for column in columns:
            before = df[column]
            after = before.mask(before.isin(to_replace), replacement)
            result[column] = after
            changes.append(changed_cells_frame(before, after, column, f"replaced {to_replace}"))
        return OperationResult(result, changed_cells=concat_changed_cells(changes))

    def to_code(
        self, to_replace: list[Any], value: Any = None, columns: list[str] | None = None
    ) -> str:
        replacement = "np.nan" if value is None else code_literal(value)
        target_columns = code_literal(columns) if columns else "df.columns"
        return (
            f"for column in {target_columns}:\n"
            f"    df[column] = df[column].mask(df[column].isin({code_literal(to_replace)}), {replacement})"
        )


class CleanText(Operation):
    name = "clean_text"
    summary = "Trim whitespace and normalize letter case in text columns."
    parameter_docs = {
        "columns": "text columns to clean",
        "strip": "trim leading/trailing whitespace (default true)",
        "collapse_spaces": "turn runs of whitespace into a single space (default false)",
        "case": "'none', 'lower', 'upper' or 'title'",
    }
    list_parameters = ("columns",)
    _CASES = ("none", "lower", "upper", "title")

    def apply(
        self,
        df: pd.DataFrame,
        columns: list[str],
        strip: bool = True,
        collapse_spaces: bool = False,
        case: str = "none",
    ) -> OperationResult:
        require_columns(df, columns)
        if case not in self._CASES:
            raise OperationError(f"case must be one of {self._CASES}")
        result, changes = df.copy(), []
        for column in columns:
            before = df[column]
            after = before.astype("string")
            if strip:
                after = after.str.strip()
            if collapse_spaces:
                after = after.str.replace(r"\s+", " ", regex=True)
            if case != "none":
                after = getattr(after.str, case)()
            result[column] = after
            changes.append(changed_cells_frame(before, after, column, "text cleaned"))
        return OperationResult(result, changed_cells=concat_changed_cells(changes))

    def to_code(
        self,
        columns: list[str],
        strip: bool = True,
        collapse_spaces: bool = False,
        case: str = "none",
    ) -> str:
        string_calls = []
        if strip:
            string_calls.append(".strip()")
        if collapse_spaces:
            string_calls.append(".replace(r'\\s+', ' ', regex=True)")
        if case != "none":
            string_calls.append(f".{case}()")
        return "\n".join(
            f"df[{code_literal(column)}] = {_chain_string_methods(column, string_calls)}"
            for column in columns
        )


def _chain_string_methods(column: str, calls: list[str]) -> str:
    """Build ``df[col].astype('string').str.a().str.b()`` from method-call fragments."""
    expression = f"df[{code_literal(column)}].astype('string')"
    for call in calls:
        expression += f".str{call}"
    return expression
