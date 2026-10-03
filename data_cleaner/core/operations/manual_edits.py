"""Operations produced by hand-editing the table: cell edits, row deletes/adds/restores."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .base import (
    CHANGED_CELLS_COLUMNS,
    Operation,
    OperationError,
    OperationResult,
    code_literal,
    removed_rows_frame,
    require_columns,
)


class SetCells(Operation):
    name = "set_cells"
    summary = "Edit individual cells (recorded from the editable table)."
    parameter_docs = {"edits": "list of {row: index label, column: name, value: new value}"}

    def apply(self, df: pd.DataFrame, edits: list[dict[str, Any]]) -> OperationResult:
        require_columns(df, [edit["column"] for edit in edits])
        result = df.copy()
        records = []
        for edit in edits:
            row, column = edit["row"], edit["column"]
            if row not in result.index:
                raise OperationError(f"Row label {row!r} does not exist.")
            records.append((row, column, result.at[row, column], edit["value"]))
            result.at[row, column] = edit["value"]
        changed = pd.DataFrame(records, columns=CHANGED_CELLS_COLUMNS[:4])
        changed["reason"] = "edited manually"
        return OperationResult(result, changed_cells=changed)

    def to_code(self, edits: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"df.at[{code_literal(e['row'])}, {code_literal(e['column'])}] = {code_literal(e['value'])}"
            for e in edits
        )


class DropRows(Operation):
    name = "drop_rows"
    summary = "Delete rows by index label (removed rows go to quarantine)."
    parameter_docs = {"labels": "index labels of the rows to delete"}
    list_parameters = ("labels",)

    def apply(self, df: pd.DataFrame, labels: list[Any]) -> OperationResult:
        is_deleted = df.index.isin(labels)
        removed = removed_rows_frame(df, is_deleted, "deleted manually")
        return OperationResult(df.loc[~is_deleted], removed_rows=removed)

    def to_code(self, labels: list[Any]) -> str:
        return f"df = df.drop(index={code_literal(labels)})"


class AddRows(Operation):
    name = "add_rows"
    summary = "Append new rows."
    parameter_docs = {"records": "list of {column: value} dicts"}

    def apply(self, df: pd.DataFrame, records: list[dict[str, Any]]) -> OperationResult:
        new_rows = pd.DataFrame(records, columns=df.columns)
        return OperationResult(pd.concat([df, new_rows], ignore_index=True))

    def to_code(self, records: list[dict[str, Any]]) -> str:
        return (
            f"new_rows = pd.DataFrame({code_literal(records)}, columns=df.columns)\n"
            "df = pd.concat([df, new_rows], ignore_index=True)"
        )


class RestoreRows(Operation):
    name = "restore_rows"
    summary = "Put quarantined rows back into the dataset."
    parameter_docs = {
        "sources": "list of [step_number, row_label] pairs identifying the quarantined rows",
        "rows": "the rows as a pandas 'split' dict (index/columns/data)",
    }

    def apply(
        self, df: pd.DataFrame, sources: list[list[Any]], rows: dict[str, Any]
    ) -> OperationResult:
        restored = pd.DataFrame(**rows)
        missing = restored.columns.difference(df.columns)
        if len(missing):
            raise OperationError(f"Restored rows have columns the data no longer has: {list(missing)}")
        # JSON turned nulls/dates into None/strings; cast back to this table's dtypes.
        restored = restored.reindex(columns=df.columns).astype(df.dtypes.to_dict(), errors="ignore")
        return OperationResult(pd.concat([df, restored]))

    def to_code(self, sources: list[list[Any]], rows: dict[str, Any]) -> str:
        return (
            f"restored_rows = pd.DataFrame(**{code_literal(rows)})\n"
            "restored_rows = restored_rows.reindex(columns=df.columns)"
            ".astype(df.dtypes.to_dict(), errors='ignore')\n"
            "df = pd.concat([df, restored_rows])"
        )


def rows_to_split_dict(rows: pd.DataFrame) -> dict[str, Any]:
    """JSON-safe 'split' representation of ``rows`` (index, columns, data)."""
    return json.loads(rows.to_json(orient="split", date_format="iso"))
