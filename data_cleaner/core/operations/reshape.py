"""Operations that change the shape of the table: group by, pivot, stack, melt.

These replace the table with a new structure, so they quarantine nothing; the
original dataset and the step history remain available. Group by and pivot are
``analysis_only``: the Analyze section runs them to produce summaries without editing
the dataset; stack and melt remain cleaning steps.
"""

from __future__ import annotations

import pandas as pd

from .base import Operation, OperationError, OperationResult, code_literal, require_columns

AGGREGATION_FUNCTIONS = ["sum", "mean", "median", "min", "max", "count", "nunique", "std", "var", "first", "last"]

# Flattens pivot_table's MultiIndex columns ("sales", "East") into "sales_East".
_FLATTEN_COLUMNS_CODE = (
    "df.columns = [\n"
    "    '_'.join(str(part) for part in column) if isinstance(column, tuple) else str(column)\n"
    "    for column in df.columns\n"
    "]"
)


def _flatten_column_name(column: object) -> str:
    if isinstance(column, tuple):
        return "_".join(str(part) for part in column)
    return str(column)


class GroupBy(Operation):
    name = "group_by"
    summary = "Group rows and aggregate columns, with an optional row count per group."
    analysis_only = True
    parameter_docs = {
        "by": "columns to group by",
        "aggregations": f"dict of {{column: [functions]}}; functions: {', '.join(AGGREGATION_FUNCTIONS)}",
        "include_row_count": "add a row_count column with the number of rows in each group",
    }
    list_parameters = ("by",)

    def apply(
        self,
        df: pd.DataFrame,
        by: list[str],
        aggregations: dict[str, list[str]] | None = None,
        include_row_count: bool = False,
    ) -> OperationResult:
        require_columns(df, by + list(aggregations or {}))
        named = self._named_aggregations(aggregations or {}, include_row_count)
        grouped = df.groupby(by)
        if not named:
            return OperationResult(grouped.size().rename("row_count").reset_index())
        result = grouped.agg(**named).reset_index()
        if include_row_count:
            result.insert(len(by), "row_count", grouped.size().to_numpy())
        return OperationResult(result)

    def to_code(
        self,
        by: list[str],
        aggregations: dict[str, list[str]] | None = None,
        include_row_count: bool = False,
    ) -> str:
        named = self._named_aggregations(aggregations or {}, include_row_count)
        lines = [f"grouped = df.groupby({code_literal(by)})"]
        if not named:
            lines.append("df = grouped.size().rename('row_count').reset_index()")
            return "\n".join(lines)
        lines.append(f"df = grouped.agg(\n    **{code_literal(named)}\n).reset_index()")
        if include_row_count:
            lines.append(f"df.insert({len(by)}, 'row_count', grouped.size().to_numpy())")
        return "\n".join(lines)

    @staticmethod
    def _named_aggregations(
        aggregations: dict[str, list[str]], include_row_count: bool
    ) -> dict[str, tuple[str, str]]:
        """{"sales": ["sum"]} -> {"sales_sum": ("sales", "sum")} (pandas named aggregation)."""
        if not aggregations and not include_row_count:
            raise OperationError("Choose columns and functions to aggregate, or include the row count.")
        named = {}
        for column, functions in aggregations.items():
            for function in functions:
                if function not in AGGREGATION_FUNCTIONS:
                    raise OperationError(f"Unknown aggregation '{function}'.")
                named[f"{column}_{function}"] = (column, function)
        return named


class Pivot(Operation):
    name = "pivot"
    summary = "Pivot: unique values of one column become new columns."
    analysis_only = True
    parameter_docs = {
        "index": "columns that stay as rows",
        "columns": "column(s) whose values become new column headers",
        "values": "column(s) to aggregate",
        "aggfunc": f"aggregation, one of {', '.join(AGGREGATION_FUNCTIONS)} (default sum)",
    }
    list_parameters = ("index", "columns", "values")

    def apply(
        self,
        df: pd.DataFrame,
        index: list[str],
        columns: list[str],
        values: list[str],
        aggfunc: str = "sum",
    ) -> OperationResult:
        require_columns(df, index + columns + values)
        if not (index and columns and values):
            raise OperationError("Pivot needs index, columns and values.")
        pivoted = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
        pivoted.columns = [_flatten_column_name(column) for column in pivoted.columns]
        return OperationResult(pivoted.reset_index())

    def to_code(
        self, index: list[str], columns: list[str], values: list[str], aggfunc: str = "sum"
    ) -> str:
        return (
            f"df = df.pivot_table(index={code_literal(index)}, columns={code_literal(columns)}, "
            f"values={code_literal(values)}, aggfunc={code_literal(aggfunc)})\n"
            f"{_FLATTEN_COLUMNS_CODE}\n"
            "df = df.reset_index()"
        )


class Stack(Operation):
    name = "stack"
    summary = "Stack: move all non-id columns into rows (wide to long)."
    parameter_docs = {
        "id_columns": "columns to keep as identifiers",
        "var_name": "name for the new column holding old column names (default 'variable')",
        "value_name": "name for the new column holding the values (default 'value')",
    }
    list_parameters = ("id_columns",)

    def apply(
        self,
        df: pd.DataFrame,
        id_columns: list[str],
        var_name: str = "variable",
        value_name: str = "value",
    ) -> OperationResult:
        require_columns(df, id_columns)
        if not id_columns or len(id_columns) == len(df.columns):
            raise OperationError("Choose id columns, leaving at least one column to stack.")
        stacked = (
            df.set_index(id_columns)
            .stack()
            .rename(value_name)
            .rename_axis(id_columns + [var_name])
            .reset_index()
        )
        return OperationResult(stacked)

    def to_code(
        self, id_columns: list[str], var_name: str = "variable", value_name: str = "value"
    ) -> str:
        return (
            f"df = (\n"
            f"    df.set_index({code_literal(id_columns)})\n"
            f"    .stack()\n"
            f"    .rename({code_literal(value_name)})\n"
            f"    .rename_axis({code_literal(id_columns)} + [{code_literal(var_name)}])\n"
            f"    .reset_index()\n)"
        )


class Melt(Operation):
    name = "melt"
    summary = "Melt: unpivot columns into variable/value rows (wide to long)."
    parameter_docs = {
        "id_vars": "columns to keep as identifiers",
        "value_vars": "columns to unpivot (default: every non-id column)",
        "var_name": "name for the column holding old column names",
        "value_name": "name for the column holding the values",
    }
    list_parameters = ("id_vars", "value_vars")

    def apply(
        self,
        df: pd.DataFrame,
        id_vars: list[str] | None = None,
        value_vars: list[str] | None = None,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> OperationResult:
        require_columns(df, (id_vars or []) + (value_vars or []))
        melted = df.melt(
            id_vars=id_vars or None,
            value_vars=value_vars or None,
            var_name=var_name,
            value_name=value_name,
        )
        return OperationResult(melted)

    def to_code(
        self,
        id_vars: list[str] | None = None,
        value_vars: list[str] | None = None,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> str:
        return (
            f"df = df.melt(id_vars={code_literal(id_vars or None)}, "
            f"value_vars={code_literal(value_vars or None)}, "
            f"var_name={code_literal(var_name)}, value_name={code_literal(value_name)})"
        )
