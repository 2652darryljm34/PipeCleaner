"""Profile a DataFrame and suggest cleaning steps, each ready to apply as-is."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from .operations import OPERATIONS

if TYPE_CHECKING:
    from .dataset import Dataset

SAMPLE_SIZE = 500  # values inspected when guessing numeric/datetime text columns
MIN_PARSE_SHARE = 0.9  # share of values that must parse to suggest a type conversion
HIGH_NULL_SHARE = 0.6  # columns with more nulls than this are suggested for removal
MAX_OUTLIER_SHARE = 0.05  # above this, "outliers" look like the distribution, not errors
CATEGORY_MAX_UNIQUE = 50
BOOLEAN_WORDS = {"true", "false", "t", "f", "yes", "no", "y", "n", "0", "1"}

HIGH, MEDIUM, LOW = 1, 2, 3


@dataclass
class Recommendation:
    """A suggested cleaning step: what to do, why, and the exact operation to run."""

    title: str
    reason: str
    operation: str
    params: dict[str, Any]
    priority: int = MEDIUM
    columns: list[str] = field(default_factory=list)

    def code(self) -> str:
        return OPERATIONS[self.operation].to_code(**self.params)

    @property
    def repeat_key(self) -> str:
        """Identity used to apply each suggestion at most once in ``apply_all_recommendations``.

        Outlier filters embed thresholds computed from the data, which shift after every
        pass, so they are identified by operation and column only.
        """
        if self.operation == "filter_rows":
            return f"{self.operation}:{self.columns}"
        return f"{self.operation}:{self.columns}:{json.dumps(self.params, sort_keys=True, default=str)}"


def recommend(df: pd.DataFrame) -> list[Recommendation]:
    """Return suggestions for ``df`` sorted by priority (most important first)."""
    if df.empty:
        return []
    checks: list[Callable[[pd.DataFrame], list[Recommendation]]] = [
        _duplicate_rows,
        _null_values,
        _constant_columns,
        _text_formatting,
        _type_conversions,
        _column_names,
        _numeric_outliers,
    ]
    recommendations = [item for check in checks for item in check(df)]
    return sorted(recommendations, key=lambda recommendation: recommendation.priority)


# ----------------------------------------------------------------------- checks


def _duplicate_rows(df: pd.DataFrame) -> list[Recommendation]:
    duplicate_count = int(df.duplicated().sum())
    if not duplicate_count:
        return []
    return [
        Recommendation(
            title=f"Drop {duplicate_count} duplicate row(s)",
            reason="Identical rows inflate counts and sums.",
            operation="drop_duplicates",
            params={"subset": None, "keep": "first"},
            priority=HIGH,
        )
    ]


def _null_values(df: pd.DataFrame) -> list[Recommendation]:
    recommendations = []
    all_null_rows = int(df.isna().all(axis=1).sum())
    if all_null_rows:
        recommendations.append(
            Recommendation(
                title=f"Drop {all_null_rows} completely empty row(s)",
                reason="These rows contain no information.",
                operation="drop_nulls",
                params={"subset": None, "how": "all"},
                priority=HIGH,
            )
        )
    for column in df.columns:
        null_share = df[column].isna().mean()
        if null_share == 0:
            continue
        percent = f"{null_share:.0%}"
        if null_share >= HIGH_NULL_SHARE:
            recommendations.append(
                Recommendation(
                    title=f"Drop mostly-empty column '{column}'",
                    reason=f"{percent} of its values are null.",
                    operation="drop_columns",
                    params={"columns": [column]},
                    priority=MEDIUM,
                    columns=[column],
                )
            )
        elif pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
            recommendations.append(
                Recommendation(
                    title=f"Fill nulls in '{column}' with the median",
                    reason=f"{percent} null; the median is robust to outliers.",
                    operation="fill_nulls",
                    params={"columns": [column], "strategy": "median"},
                    priority=MEDIUM,
                    columns=[column],
                )
            )
        elif df[column].dropna().size:
            recommendations.append(
                Recommendation(
                    title=f"Fill nulls in '{column}' with the most common value",
                    reason=f"{percent} null. Alternatively drop those rows.",
                    operation="fill_nulls",
                    params={"columns": [column], "strategy": "mode"},
                    priority=LOW,
                    columns=[column],
                )
            )
    return recommendations


def _constant_columns(df: pd.DataFrame) -> list[Recommendation]:
    if len(df) < 2:
        return []
    return [
        Recommendation(
            title=f"Drop constant column '{column}'",
            reason="It has a single value for every row, so it carries no information.",
            operation="drop_columns",
            params={"columns": [column]},
            priority=LOW,
            columns=[column],
        )
        for column in df.columns
        if df[column].nunique(dropna=True) == 1 and not df[column].isna().any()
    ]


def _text_columns(df: pd.DataFrame) -> list[str]:
    """Columns holding text. Categoricals are excluded: pandas calls them string-like, but
    they are already converted, and suggesting category -> category again never ends."""
    return [
        column
        for column in df.columns
        if not isinstance(df[column].dtype, pd.CategoricalDtype)
        and (pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]))
    ]


def _text_formatting(df: pd.DataFrame) -> list[Recommendation]:
    recommendations = []
    for column in _text_columns(df):
        text = df[column].dropna().astype("string")
        if text.empty:
            continue
        if (text != text.str.strip()).any():
            recommendations.append(
                Recommendation(
                    title=f"Trim whitespace in '{column}'",
                    reason="Some values have leading/trailing spaces, so 'a' and 'a ' differ.",
                    operation="clean_text",
                    params={"columns": [column], "strip": True, "collapse_spaces": False, "case": "none"},
                    priority=HIGH,
                    columns=[column],
                )
            )
        if text.str.strip().str.lower().nunique() < text.str.strip().nunique():
            recommendations.append(
                Recommendation(
                    title=f"Standardize letter case in '{column}'",
                    reason="Values differ only by capitalization (e.g. 'Paris' vs 'paris').",
                    operation="clean_text",
                    params={"columns": [column], "strip": True, "collapse_spaces": False, "case": "lower"},
                    priority=MEDIUM,
                    columns=[column],
                )
            )
    return recommendations


def _type_conversions(df: pd.DataFrame) -> list[Recommendation]:
    recommendations = []
    for column in _text_columns(df):
        values = df[column].dropna().astype("string").str.strip()
        if values.empty:
            continue
        sample = values.head(SAMPLE_SIZE)

        numeric = pd.to_numeric(sample, errors="coerce")
        if numeric.notna().mean() >= MIN_PARSE_SHARE:
            all_integers = bool((numeric.dropna() % 1 == 0).all())
            recommendations.append(
                _conversion(column, "int" if all_integers else "float", numeric.notna().mean())
            )
            continue

        if set(sample.str.lower().unique()) <= BOOLEAN_WORDS and sample.nunique() >= 2:
            recommendations.append(_conversion(column, "bool", 1.0))
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed_dates = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed_dates.notna().mean() >= MIN_PARSE_SHARE:
            recommendations.append(_conversion(column, "datetime", parsed_dates.notna().mean()))
            continue

        unique_count = values.nunique()
        if len(values) >= 100 and unique_count <= CATEGORY_MAX_UNIQUE and unique_count / len(values) < 0.05:
            recommendations.append(
                Recommendation(
                    title=f"Convert '{column}' to category",
                    reason=f"Only {unique_count} distinct values across {len(values)} rows; saves memory.",
                    operation="convert_dtype",
                    params={"column": column, "dtype": "category"},
                    priority=LOW,
                    columns=[column],
                )
            )
    return recommendations


def _conversion(column: str, dtype: str, parse_share: float) -> Recommendation:
    note = "" if parse_share == 1 else " Values that do not parse will be quarantined as nulls."
    return Recommendation(
        title=f"Convert '{column}' to {dtype}",
        reason=f"Stored as text, but {parse_share:.0%} of values look like {dtype} values.{note}",
        operation="convert_dtype",
        params={"column": column, "dtype": dtype},
        priority=HIGH,
        columns=[column],
    )


def _column_names(df: pd.DataFrame) -> list[Recommendation]:
    """Suggest snake_case names when they would not collide with each other."""
    mapping = {column: _snake_case(str(column)) for column in df.columns}
    mapping = {old: new for old, new in mapping.items() if new and new != old}
    final_names = [mapping.get(column, column) for column in df.columns]
    if not mapping or len(set(final_names)) != len(final_names):
        return []
    return [
        Recommendation(
            title=f"Rename {len(mapping)} column(s) to snake_case",
            reason="Spaces, capitals and symbols in names make code and queries awkward.",
            operation="rename_columns",
            params={"mapping": mapping},
            priority=LOW,
            columns=list(mapping),
        )
    ]


def _snake_case(name: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)  # camelCase -> camel_Case
    return re.sub(r"[^0-9a-zA-Z]+", "_", spaced).strip("_").lower()


def _numeric_outliers(df: pd.DataFrame) -> list[Recommendation]:
    """Suggest IQR-fence filters. Only for columns without nulls, since filters drop nulls."""
    recommendations = []
    for column in df.columns:
        series = df[column]
        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or series.isna().any()
            or series.nunique() < 10
        ):
            continue
        first_quartile, third_quartile = series.quantile([0.25, 0.75])
        spread = third_quartile - first_quartile
        lower, upper = first_quartile - 1.5 * spread, third_quartile + 1.5 * spread
        outlier_count = int(((series < lower) | (series > upper)).sum())
        if not outlier_count or outlier_count / len(series) > MAX_OUTLIER_SHARE:
            continue
        recommendations.append(
            Recommendation(
                title=f"Review {outlier_count} outlier(s) in '{column}'",
                reason=f"Outside the IQR fences [{lower:.4g}, {upper:.4g}]. Rows are quarantined, not lost.",
                operation="filter_rows",
                params={
                    "conditions": [
                        {"column": column, "operator": ">=", "value": float(lower)},
                        {"column": column, "operator": "<=", "value": float(upper)},
                    ],
                    "combine": "and",
                },
                priority=LOW,
                columns=[column],
            )
        )
    return recommendations


# --------------------------------------------------------------------- apply all


@dataclass
class ApplyAllResult:
    """What ``apply_all_recommendations`` did."""

    applied: list[str] = field(default_factory=list)  # titles of applied suggestions
    skipped: list[str] = field(default_factory=list)  # "title: reason" for suggestions that failed


def apply_all_recommendations(dataset: Dataset, max_steps: int = 100) -> ApplyAllResult:
    """Apply suggestions one at a time, re-profiling the table after each step.

    Re-profiling matters because every step changes the table (renamed or dropped columns,
    new dtypes), which would make a precomputed list stale. Each suggestion is applied at
    most once, so the loop always ends; ones that fail are skipped and reported.
    """
    result, handled = ApplyAllResult(), set()
    for _ in range(max_steps):
        pending = [item for item in recommend(dataset.current) if item.repeat_key not in handled]
        if not pending:
            break
        item = pending[0]
        handled.add(item.repeat_key)
        try:
            dataset.apply(item.operation, **item.params)
        except Exception as error:  # one bad suggestion should not stop the rest
            result.skipped.append(f"{item.title}: {error}")
        else:
            result.applied.append(item.title)
    return result
