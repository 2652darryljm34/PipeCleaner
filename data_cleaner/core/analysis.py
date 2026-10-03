"""Read-only analysis of a table: column overview, statistics, value counts, group by, pivot.

Nothing here modifies a dataset. Every function returns an ``AnalysisResult``: the result
table plus the pandas code that produces it, so analysis is as transparent as cleaning.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .operations import OPERATIONS
from .operations.base import code_literal, require_columns

NUMERIC_STATISTICS = ["count", "sum", "mean", "median", "std", "var", "min", "q1", "q3", "max"]


@dataclass
class AnalysisResult:
    """A result table and the pandas code that produced it (reading ``df``)."""

    table: pd.DataFrame
    code: str


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric, non-boolean columns (the ones sums, means and std dev make sense for)."""
    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column])
    ]


def column_overview(df: pd.DataFrame, columns: list[str] | None = None) -> AnalysisResult:
    """One row per column: type, non-null count, nulls, distinct values, most common value."""
    columns = columns or list(df.columns)
    require_columns(df, columns)
    selected = df[columns]
    overview = pd.DataFrame(
        {
            "dtype": selected.dtypes.astype(str),
            "non_null": selected.count(),
            "nulls": selected.isna().sum(),
            "null_pct": (selected.isna().mean() * 100).round(2),
            "distinct": selected.nunique(),
            # Strings keep mixed-type columns displayable.
            "most_common": selected.apply(lambda s: str(s.mode().iloc[0]) if s.notna().any() else ""),
        }
    )
    code = textwrap.dedent(
        f"""\
        selected = df[{code_literal(columns)}]
        overview = pd.DataFrame({{
            'dtype': selected.dtypes.astype(str),
            'non_null': selected.count(),
            'nulls': selected.isna().sum(),
            'null_pct': (selected.isna().mean() * 100).round(2),
            'distinct': selected.nunique(),
            'most_common': selected.apply(lambda s: str(s.mode().iloc[0]) if s.notna().any() else ''),
        }})"""
    )
    return AnalysisResult(overview, code)


def numeric_statistics(df: pd.DataFrame, columns: list[str] | None = None) -> AnalysisResult:
    """One row per numeric column: count, sum, mean, median, std, variance, min, quartiles, max."""
    columns = columns or numeric_columns(df)
    require_columns(df, columns)
    numeric = df[columns]
    statistics = numeric.agg(["count", "sum", "mean", "median", "std", "var", "min", "max"]).T
    quartiles = numeric.quantile([0.25, 0.75]).T.set_axis(["q1", "q3"], axis=1)
    statistics = statistics.join(quartiles)[NUMERIC_STATISTICS]
    code = textwrap.dedent(
        f"""\
        numeric = df[{code_literal(columns)}]
        statistics = numeric.agg(['count', 'sum', 'mean', 'median', 'std', 'var', 'min', 'max']).T
        quartiles = numeric.quantile([0.25, 0.75]).T.set_axis(['q1', 'q3'], axis=1)
        statistics = statistics.join(quartiles)[{code_literal(NUMERIC_STATISTICS)}]"""
    )
    return AnalysisResult(statistics, code)


def value_counts(
    df: pd.DataFrame, column: str, include_nulls: bool = False, limit: int | None = None
) -> AnalysisResult:
    """Distinct values of ``column`` with their counts and share of rows, most frequent first."""
    require_columns(df, [column])
    counts = df[column].value_counts(dropna=not include_nulls).rename_axis("value").reset_index(name="count")
    counts["percent"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    if limit:
        counts = counts.head(limit)
    drop_nulls = not include_nulls
    code = (
        f"counts = df[{code_literal(column)}].value_counts(dropna={drop_nulls}).rename_axis('value').reset_index(name='count')\n"
        "counts['percent'] = (counts['count'] / counts['count'].sum() * 100).round(2)"
    )
    if limit:
        code += f"\ncounts = counts.head({limit})"
    return AnalysisResult(counts, code)


def correlations(df: pd.DataFrame, method: str = "pearson") -> AnalysisResult:
    """Correlation matrix of the numeric columns."""
    table = df.corr(numeric_only=True, method=method)
    return AnalysisResult(table, f"correlations = df.corr(numeric_only=True, method={method!r})")


def grouped_summary(
    df: pd.DataFrame,
    by: list[str],
    aggregations: dict[str, list[str]],
    include_row_count: bool = True,
) -> AnalysisResult:
    """Aggregate columns per group (e.g. sum and mean of sales per region)."""
    return _run_analysis_operation(
        df, "group_by", {"by": by, "aggregations": aggregations, "include_row_count": include_row_count}
    )


def pivot_summary(
    df: pd.DataFrame, index: list[str], columns: list[str], values: list[str], aggfunc: str = "sum"
) -> AnalysisResult:
    """Cross-tabulate: rows from ``index``, columns from ``columns``, cells aggregate ``values``."""
    return _run_analysis_operation(
        df, "pivot", {"index": index, "columns": columns, "values": values, "aggfunc": aggfunc}
    )


def _run_analysis_operation(df: pd.DataFrame, operation_name: str, params: dict[str, Any]) -> AnalysisResult:
    """Run an analysis-only operation on ``df`` without touching it, and wrap its code safely.

    The operation's code rebinds ``df``, so it is wrapped in a function: ``df`` inside is a
    local name and the caller's table stays untouched.
    """
    operation = OPERATIONS[operation_name]
    normalized = operation.normalize(df, params)
    table = operation.apply(df, **normalized).data
    body = textwrap.indent(operation.to_code(**normalized), "    ")
    code = f"def summarize(df):\n{body}\n    return df\n\n\nsummary = summarize(df)  # analysis only: df itself is unchanged"
    return AnalysisResult(table, code)
