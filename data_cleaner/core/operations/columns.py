"""Operations that add, remove, derive and order columns."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Operation, OperationError, OperationResult, Params, code_literal, require_columns


class DropColumns(Operation):
    name = "drop_columns"
    summary = "Remove columns (the original dataset still has them)."
    parameter_docs = {"columns": "columns to remove"}
    list_parameters = ("columns",)

    def apply(self, df: pd.DataFrame, columns: list[str]) -> OperationResult:
        require_columns(df, columns)
        return OperationResult(df.drop(columns=columns))

    def to_code(self, columns: list[str]) -> str:
        return f"df = df.drop(columns={code_literal(columns)})"


class AddColumn(Operation):
    name = "add_column"
    summary = "Add a column from an expression (e.g. 'price * quantity') or a constant."
    parameter_docs = {
        "name": "new column name",
        "mode": "'expression' (pandas eval over existing columns) or 'constant'",
        "expression": "expression text, e.g. `price * quantity` (mode=expression)",
        "value": "constant value (mode=constant)",
    }

    def apply(
        self,
        df: pd.DataFrame,
        name: str,
        mode: str = "expression",
        expression: str | None = None,
        value: Any = None,
    ) -> OperationResult:
        if not name:
            raise OperationError("The new column needs a name.")
        result = df.copy()
        if mode == "expression":
            if not expression:
                raise OperationError("Enter an expression.")
            try:
                result[name] = df.eval(expression)
            except Exception as error:  # pandas raises many types for bad expressions
                raise OperationError(f"Invalid expression '{expression}': {error}") from error
        elif mode == "constant":
            result[name] = value
        else:
            raise OperationError("mode must be 'expression' or 'constant'")
        return OperationResult(result)

    def to_code(
        self, name: str, mode: str = "expression", expression: str | None = None, value: Any = None
    ) -> str:
        if mode == "expression":
            return f"df[{code_literal(name)}] = df.eval({code_literal(expression)})"
        return f"df[{code_literal(name)}] = {code_literal(value)}"


class CreateBins(Operation):
    name = "create_bins"
    summary = "Bin a numeric column into a new categorical column."
    parameter_docs = {
        "column": "numeric column to bin",
        "method": "'equal_width' (bins=int), 'quantile' (bins=int) or 'custom' (bins=list of edges)",
        "bins": "number of bins, or a list of bin edges for 'custom'",
        "labels": "optional list of labels (one per bin)",
        "new_column": "name of the new column (default: <column>_binned)",
        "right": "whether bins include their right edge (default true)",
    }
    list_parameters = ("labels",)
    _METHODS = ("equal_width", "quantile", "custom")

    def normalize(self, df: pd.DataFrame, params: Params) -> Params:
        normalized = super().normalize(df, params)
        if normalized.get("method") == "custom":
            edges = normalized["bins"]
            if isinstance(edges, str):
                edges = [part for part in edges.split(",") if part.strip()]
            normalized["bins"] = [float(edge) for edge in edges]
        else:
            normalized["bins"] = int(normalized["bins"])
        if not normalized.get("labels"):
            normalized["labels"] = None
        normalized.setdefault("new_column", f"{normalized['column']}_binned")
        return normalized

    def apply(
        self,
        df: pd.DataFrame,
        column: str,
        method: str,
        bins: int | list[float],
        new_column: str,
        labels: list[str] | None = None,
        right: bool = True,
    ) -> OperationResult:
        require_columns(df, [column])
        if method not in self._METHODS:
            raise OperationError(f"method must be one of {self._METHODS}")
        values = pd.to_numeric(df[column], errors="coerce")
        result = df.copy()
        try:
            if method == "quantile":
                result[new_column] = pd.qcut(values, q=bins, labels=labels, duplicates="drop")
            else:
                result[new_column] = pd.cut(
                    values, bins=bins, labels=labels, right=right, include_lowest=True
                )
        except ValueError as error:
            raise OperationError(f"Could not bin '{column}': {error}") from error
        return OperationResult(result)

    def to_code(
        self,
        column: str,
        method: str,
        bins: int | list[float],
        new_column: str,
        labels: list[str] | None = None,
        right: bool = True,
    ) -> str:
        values = f"pd.to_numeric(df[{code_literal(column)}], errors='coerce')"
        target = f"df[{code_literal(new_column)}]"
        if method == "quantile":
            return f"{target} = pd.qcut({values}, q={bins}, labels={code_literal(labels)}, duplicates='drop')"
        return (
            f"{target} = pd.cut({values}, bins={code_literal(bins)}, "
            f"labels={code_literal(labels)}, right={right}, include_lowest=True)"
        )


class SortValues(Operation):
    name = "sort_values"
    summary = "Sort rows by one or more columns."
    parameter_docs = {"by": "columns to sort by", "ascending": "true (default) or false"}
    list_parameters = ("by",)

    def apply(self, df: pd.DataFrame, by: list[str], ascending: bool = True) -> OperationResult:
        require_columns(df, by)
        return OperationResult(df.sort_values(by=by, ascending=ascending))

    def to_code(self, by: list[str], ascending: bool = True) -> str:
        return f"df = df.sort_values(by={code_literal(by)}, ascending={ascending})"
