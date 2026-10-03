"""Registry of every available cleaning operation."""

from __future__ import annotations

from .base import Operation, OperationError, OperationResult, Params, parse_cli_value, parse_scalar
from .cleaning import (
    CleanText,
    ConvertDType,
    DropDuplicates,
    DropNulls,
    FillNulls,
    FilterRows,
    RenameColumns,
    ReplaceValues,
)
from .columns import AddColumn, CreateBins, DropColumns, SortValues
from .manual_edits import AddRows, DropRows, RestoreRows, SetCells
from .reshape import GroupBy, Melt, Pivot, Stack

_ALL_OPERATIONS: list[Operation] = [
    RenameColumns(), DropDuplicates(), DropNulls(), FilterRows(), ConvertDType(),
    FillNulls(), ReplaceValues(), CleanText(), CreateBins(), AddColumn(),
    DropColumns(), SortValues(), GroupBy(), Pivot(), Stack(), Melt(),
    SetCells(), DropRows(), AddRows(), RestoreRows(),
]  # fmt: skip

OPERATIONS: dict[str, Operation] = {operation.name: operation for operation in _ALL_OPERATIONS}


def get_operation(name: str) -> Operation:
    """Look up an operation by name, with a helpful error for typos."""
    try:
        return OPERATIONS[name]
    except KeyError:
        raise OperationError(f"Unknown operation '{name}'. Available: {sorted(OPERATIONS)}") from None


__all__ = [
    "OPERATIONS",
    "Operation",
    "OperationError",
    "OperationResult",
    "Params",
    "get_operation",
    "parse_cli_value",
    "parse_scalar",
]
