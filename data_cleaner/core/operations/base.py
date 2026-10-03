"""Foundations shared by every cleaning operation.

An *operation* is a stateless object with two jobs:

* ``apply``   - transform a DataFrame with pandas and report what was quarantined.
* ``to_code`` - return the equivalent pandas snippet, operating on a variable ``df``.

Because operations are driven purely by a JSON-friendly ``params`` dict, a list of
steps can be replayed, edited, saved to disk, and exported as a standalone script.
"""

from __future__ import annotations

import ast
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

Params = dict[str, Any]

# Columns that describe *why* a row/cell was quarantined; data columns follow them.
REMOVED_REASON_COLUMN = "_reason"
CHANGED_CELLS_COLUMNS = ["row", "column", "before", "after", "reason"]


class OperationError(ValueError):
    """Raised when an operation cannot run with the parameters it was given."""


@dataclass
class OperationResult:
    """The outcome of applying one operation."""

    data: pd.DataFrame
    # Rows that left the dataset, with an extra ``_reason`` column. Index labels are kept.
    removed_rows: pd.DataFrame | None = None
    # One row per modified cell: row label, column, value before, value after, reason.
    changed_cells: pd.DataFrame | None = None


class Operation(ABC):
    """Base class for all cleaning operations."""

    name: ClassVar[str]
    summary: ClassVar[str]
    # Human-readable description of each parameter (used by CLI help).
    parameter_docs: ClassVar[dict[str, str]] = {}
    # Parameters that must be lists; comma-separated strings are split automatically.
    list_parameters: ClassVar[tuple[str, ...]] = ()

    def normalize(self, df: pd.DataFrame, params: Params) -> Params:
        """Return a canonical copy of ``params`` (coerced types, lists, ...).

        Called before a step is stored so that the stored params, the generated code
        and ``apply`` all agree. ``df`` is the frame the step will run against.
        """
        normalized = dict(params)
        for key in self.list_parameters:
            if key in normalized:
                normalized[key] = as_list(normalized[key])
        return normalized

    @abstractmethod
    def apply(self, df: pd.DataFrame, **params: Any) -> OperationResult:
        """Run the operation. Must not mutate ``df``."""

    @abstractmethod
    def to_code(self, **params: Any) -> str:
        """Return pandas code equivalent to ``apply`` (reads and writes ``df``)."""


# --------------------------------------------------------------------------- helpers


def as_list(value: Any) -> list | None:
    """Coerce ``value`` to a list; ``None`` stays ``None``; "a, b" becomes ["a", "b"]."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


def parse_scalar(text: str) -> Any:
    """Turn user-typed text into int / float / bool / None / str (in that order)."""
    stripped = text.strip()
    if stripped.lower() in {"none", "null", "nan"}:
        return None
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    for parser in (int, float):
        try:
            return parser(stripped)
        except ValueError:
            continue
    return text


def parse_cli_value(text: str) -> Any:
    """Parse a ``key=value`` value from the CLI: JSON, then a Python literal, else text."""
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError):
            continue
    return text


def require_columns(df: pd.DataFrame, columns: list[str] | None) -> None:
    """Raise a helpful error if any of ``columns`` is missing from ``df``."""
    missing = [column for column in (columns or []) if column not in df.columns]
    if missing:
        raise OperationError(f"Unknown column(s): {missing}. Available: {list(df.columns)}")


def removed_rows_frame(df: pd.DataFrame, mask: Any, reason: str | list[str]) -> pd.DataFrame:
    """Rows of ``df`` selected by ``mask``, tagged with the reason they were removed."""
    removed = df.loc[np.asarray(mask, dtype=bool)].copy()
    removed.insert(0, REMOVED_REASON_COLUMN, reason)
    return removed


def changed_cells_frame(
    before: pd.Series, after: pd.Series, column: str, reason: str
) -> pd.DataFrame:
    """Describe every cell where ``after`` differs from ``before`` (NaN == NaN).

    Values are compared as strings so that 1 -> 1.0 representation changes do not
    count, while null -> value and value -> different value do.
    """
    value_changed = before.astype("string").ne(after.astype("string")).fillna(False)
    null_changed = before.isna().to_numpy() != after.isna().to_numpy()
    changed = np.asarray(value_changed, dtype=bool) | null_changed
    return pd.DataFrame(
        {
            "row": before.index[changed],
            "column": column,
            "before": before[changed].to_numpy(dtype=object),
            "after": after[changed].to_numpy(dtype=object),
            "reason": reason,
        },
        columns=CHANGED_CELLS_COLUMNS,
    )


def concat_changed_cells(frames: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Combine per-column change records; ``None`` when nothing changed."""
    non_empty = [frame for frame in frames if not frame.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else None


def code_literal(value: Any) -> str:
    """Python source text for a JSON-style value."""
    return repr(value)
