"""Plotly Express plotting driven by a declarative table of plot kinds.

Each plot kind lists the arguments it accepts, with a type per argument (column, list of
columns, number, toggle, choice). The Streamlit UI and the CLI both render their inputs
from that table, and ``plot_code`` generates the matching code.

``color``, ``size`` and ``symbol`` accept either a column (mapped per row) or a fixed value
for every point (e.g. color "red"). Fixed values are stored as ``<name>_fixed`` and applied
with ``fig.update_traces`` because Plotly Express itself only maps columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_MAX_POINTS = 50_000

DATA, STYLE = "data", "style"  # argument groups: what to plot vs how it looks

TEMPLATES = ("plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "presentation")
DISCRETE_PALETTES = (
    "Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Light24",
    "Set1", "Set2", "Set3", "Pastel", "Bold", "Safe", "Vivid", "Antique", "Prism",
)  # fmt: skip
CONTINUOUS_SCALES = (
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo",
    "Blues", "Greens", "Reds", "YlOrRd", "RdBu", "Spectral",
)  # fmt: skip
MARKER_SYMBOLS = ("circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "star", "hexagon")


@dataclass(frozen=True)
class PlotArgument:
    """One input a plot kind accepts."""

    name: str
    kind: str  # "column" | "columns" | "number" | "boolean" | "choice"
    required: bool = False
    choices: tuple[str, ...] = ()
    help: str = ""
    group: str = DATA
    number_type: str = "int"  # "int" | "float" (kind == "number")
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    # When set ("color" | "size" | "symbol"), the argument also accepts one fixed value for all points.
    fixed: str | None = None


def fixed_key(name: str) -> str:
    """Argument name under which the fixed (non-column) value of ``name`` is stored."""
    return f"{name}_fixed"


def _column(name: str, required: bool = False, help: str = "", fixed: str | None = None) -> PlotArgument:
    return PlotArgument(name, "column", required, help=help, fixed=fixed)


def _columns(name: str, required: bool = False, help: str = "") -> PlotArgument:
    return PlotArgument(name, "columns", required, help=help)


def _choice(name: str, choices: tuple[str, ...], help: str = "", group: str = STYLE) -> PlotArgument:
    return PlotArgument(name, "choice", choices=choices, help=help, group=group)


def _toggle(name: str, help: str = "", group: str = STYLE) -> PlotArgument:
    return PlotArgument(name, "boolean", help=help, group=group)


def _int(name: str, help: str = "", minimum: int = 1, group: str = DATA) -> PlotArgument:
    return PlotArgument(name, "number", help=help, group=group, minimum=minimum, step=1)


def _float(name: str, minimum: float, maximum: float, step: float, help: str = "") -> PlotArgument:
    return PlotArgument(name, "number", help=help, group=STYLE, number_type="float", minimum=minimum, maximum=maximum, step=step)


COLOR = _column("color", help="A column to color by, or a fixed color for every point", fixed="color")
FACET_COL = _column("facet_col", help="One subplot per value (columns)")
FACET_ROW = _column("facet_row", help="One subplot per value (rows)")
HOVER = _column("hover_name", help="Bold title in the hover tooltip")

TEMPLATE = _choice("template", TEMPLATES, "Overall look")
PALETTE = _choice("color_discrete_sequence", DISCRETE_PALETTES, "Colors used for categories")
SCALE = _choice("color_continuous_scale", CONTINUOUS_SCALES, "Colors used for numeric values")
OPACITY = _float("opacity", 0.0, 1.0, 0.05, "0 = invisible, 1 = solid")
LOG_X = _toggle("log_x", "Logarithmic x axis")
LOG_Y = _toggle("log_y", "Logarithmic y axis")
ORIENTATION = _choice("orientation", ("v", "h"), "v = vertical, h = horizontal")
TEXT_AUTO = _toggle("text_auto", "Print values on the marks")
POINTS = _choice("points", ("all", "outliers", "suspectedoutliers"), "Which individual points to draw")

# plot kind -> accepted arguments
PLOT_KINDS: dict[str, list[PlotArgument]] = {
    "scatter": [
        _column("x", True), _column("y", True), COLOR,
        _column("size", help="A numeric column for marker size, or a fixed size", fixed="size"),
        _column("symbol", help="A column for marker shape, or a fixed shape", fixed="symbol"),
        FACET_COL, FACET_ROW, HOVER,
        OPACITY, _choice("marginal_x", ("histogram", "rug", "box", "violin")),
        _choice("marginal_y", ("histogram", "rug", "box", "violin")), LOG_X, LOG_Y, PALETTE, SCALE, TEMPLATE,
    ],
    "line": [_column("x", True), _column("y", True), COLOR, FACET_COL, FACET_ROW, LOG_X, LOG_Y, PALETTE, TEMPLATE],
    "bar": [
        _column("x", True), _column("y", True), COLOR, FACET_COL,
        _choice("barmode", ("relative", "group", "overlay"), "How bars of different colors are arranged", group=DATA),
        _choice("barnorm", ("fraction", "percent"), "Normalize stacked bars"),
        ORIENTATION, OPACITY, TEXT_AUTO, LOG_Y, PALETTE, SCALE, TEMPLATE,
    ],
    "histogram": [
        _column("x", True), _column("y"), COLOR, FACET_COL,
        _int("nbins", "Number of bins (leave empty for automatic)"),
        PlotArgument("bin_size", "number", help="Width of each bin, in the x unit (overrides nbins)", number_type="float", minimum=0.0),
        _choice("histfunc", ("count", "sum", "avg", "min", "max"), "How to aggregate y in each bin", group=DATA),
        _choice("histnorm", ("percent", "probability", "density", "probability density")),
        _choice("barmode", ("relative", "group", "overlay"), group=DATA),
        _choice("marginal", ("rug", "box", "violin")), _toggle("cumulative"), ORIENTATION, OPACITY, TEXT_AUTO,
        LOG_X, LOG_Y, PALETTE, TEMPLATE,
    ],
    "box": [_column("x"), _column("y", True), COLOR, FACET_COL, POINTS, _toggle("notched"), ORIENTATION, LOG_Y, PALETTE, TEMPLATE],
    "violin": [_column("x"), _column("y", True), COLOR, FACET_COL, POINTS, _toggle("box", "Draw a box inside the violin"), ORIENTATION, PALETTE, TEMPLATE],
    "strip": [_column("x"), _column("y", True), COLOR, FACET_COL, ORIENTATION, LOG_Y, PALETTE, TEMPLATE],
    "area": [
        _column("x", True), _column("y", True), COLOR, FACET_COL,
        _choice("groupnorm", ("fraction", "percent"), "Normalize stacked areas"), LOG_Y, PALETTE, TEMPLATE,
    ],
    "pie": [
        _column("names", True, "Slice labels"), _column("values", False, "Slice sizes (default: row count)"),
        _float("hole", 0.0, 0.9, 0.05, "Donut hole size (0 = full pie)"), PALETTE, TEMPLATE,
    ],
    "density_heatmap": [
        _column("x", True), _column("y", True), _column("z", help="Value to aggregate"), FACET_COL,
        _int("nbinsx", "Bins along x"), _int("nbinsy", "Bins along y"),
        _choice("histfunc", ("count", "sum", "avg", "min", "max"), group=DATA), TEXT_AUTO, LOG_X, LOG_Y, SCALE, TEMPLATE,
    ],
    "density_contour": [_column("x", True), _column("y", True), COLOR, FACET_COL, _int("nbinsx"), _int("nbinsy"), PALETTE, TEMPLATE],
    "ecdf": [
        _column("x", True), COLOR, FACET_COL,
        _choice("ecdfnorm", ("probability", "percent")), _choice("ecdfmode", ("standard", "complementary", "reversed")),
        LOG_X, PALETTE, TEMPLATE,
    ],
    "scatter_3d": [
        _column("x", True), _column("y", True), _column("z", True), COLOR,
        _column("size", help="A numeric column for marker size, or a fixed size", fixed="size"),
        OPACITY, PALETTE, SCALE, TEMPLATE,
    ],
    "scatter_matrix": [_columns("dimensions", True, "Columns to compare pairwise"), COLOR, OPACITY, PALETTE, SCALE, TEMPLATE],
    "treemap": [_columns("path", True, "Hierarchy, outermost first"), _column("values"), COLOR, PALETTE, SCALE, TEMPLATE],
    "sunburst": [_columns("path", True, "Hierarchy, innermost first"), _column("values"), COLOR, PALETTE, SCALE, TEMPLATE],
    "funnel": [_column("x", True), _column("y", True), COLOR, PALETTE, TEMPLATE],
    "correlation_heatmap": [
        _choice("method", ("pearson", "spearman", "kendall"), "How correlation is measured", group=DATA),
        SCALE, TEMPLATE,
    ],
}

# Arguments that are not Plotly Express keywords; applied after the figure is built.
_POST_BUILD_ARGUMENTS = {"bin_size", "barnorm"}  # bin_size -> trace setting, barnorm -> layout setting
_LINE_KINDS = {"line", "area", "ecdf"}


# ------------------------------------------------------------------ argument handling


def resolve_fixed_arguments(df: pd.DataFrame, kind: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Treat a ``color``/``size``/``symbol`` value that is not a column as a fixed value.

    Lets CLI users write ``color=red`` or ``size=12`` instead of ``color_fixed=red``.
    """
    resolved = dict(arguments)
    for spec in PLOT_KINDS[kind]:
        value = resolved.get(spec.name)
        if spec.fixed and value is not None and value not in df.columns:
            resolved[fixed_key(spec.name)] = resolved.pop(spec.name)
    return resolved


def _clean_arguments(kind: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep only arguments this kind accepts, dropping empty values; validate required ones.

    A fixed value is dropped when the same argument is mapped to a column (the column wins).
    """
    specs = {spec.name: spec for spec in PLOT_KINDS[kind]}
    accepted = set(specs) | {fixed_key(spec.name) for spec in specs.values() if spec.fixed}
    cleaned = {
        name: value
        for name, value in arguments.items()
        if name in accepted and value is not False and value not in (None, "", [], ())
    }
    for spec in specs.values():
        if spec.fixed and spec.name in cleaned:
            cleaned.pop(fixed_key(spec.name), None)
        if spec.kind == "number" and spec.name in cleaned:
            cleaned[spec.name] = int(cleaned[spec.name]) if spec.number_type == "int" else float(cleaned[spec.name])
    missing = [spec.name for spec in specs.values() if spec.required and spec.name not in cleaned]
    if missing:
        raise ValueError(f"'{kind}' plots need: {', '.join(missing)}")
    return cleaned


def _split_arguments(cleaned: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Separate Plotly Express keywords from fixed values and post-build settings."""
    fixed = {name.removesuffix("_fixed"): value for name, value in cleaned.items() if name.endswith("_fixed")}
    post_build = {name: value for name, value in cleaned.items() if name in _POST_BUILD_ARGUMENTS}
    px_arguments = {
        name: value
        for name, value in cleaned.items()
        if not name.endswith("_fixed") and name not in _POST_BUILD_ARGUMENTS
    }
    return px_arguments, fixed, post_build


def _trace_updates(kind: str, fixed: dict[str, Any], post_build: dict[str, Any], orientation: str | None) -> dict[str, Any]:
    """``fig.update_traces`` keywords for fixed colors/sizes/symbols and histogram bin size."""
    updates: dict[str, Any] = {}
    if "color" in fixed:
        if kind in _LINE_KINDS:
            updates["line_color"] = fixed["color"]
        elif kind in {"box", "violin"}:
            updates.update(marker_color=fixed["color"], line_color=fixed["color"])
        else:
            updates["marker_color"] = fixed["color"]
    if "size" in fixed:
        updates["marker_size"] = fixed["size"]
    if "symbol" in fixed:
        updates["marker_symbol"] = fixed["symbol"]
    if "bin_size" in post_build:
        updates["ybins_size" if orientation == "h" else "xbins_size"] = post_build["bin_size"]
    return updates


def _px_values(px_arguments: dict[str, Any]) -> dict[str, Any]:
    """Resolve palette names to the color lists Plotly Express expects."""
    values = dict(px_arguments)
    if "color_discrete_sequence" in values:
        values["color_discrete_sequence"] = getattr(px.colors.qualitative, values["color_discrete_sequence"])
    return values


def _px_code(px_arguments: dict[str, Any]) -> str:
    """Source text for the Plotly Express keyword arguments."""
    parts = []
    for name, value in px_arguments.items():
        text = f"px.colors.qualitative.{value}" if name == "color_discrete_sequence" else repr(value)
        parts.append(f"{name}={text}")
    return ", ".join(parts)


def _plot_frame(df: pd.DataFrame, max_points: int | None) -> pd.DataFrame:
    """Sample large frames so interactive plots stay responsive."""
    if max_points and len(df) > max_points:
        return df.sample(max_points, random_state=0)
    return df


# --------------------------------------------------------------------- public API


def build_figure(
    df: pd.DataFrame,
    kind: str,
    arguments: dict[str, Any],
    title: str | None = None,
    max_points: int | None = DEFAULT_MAX_POINTS,
) -> go.Figure:
    """Create a Plotly figure of the given kind."""
    if kind not in PLOT_KINDS:
        raise ValueError(f"Unknown plot kind '{kind}'. Choose from {list(PLOT_KINDS)}")
    px_arguments, fixed, post_build = _split_arguments(_clean_arguments(kind, arguments))
    if kind == "correlation_heatmap":
        method = px_arguments.pop("method", "pearson")
        correlations = df.corr(numeric_only=True, method=method)
        return px.imshow(correlations, text_auto=".2f", aspect="auto", title=title or "Correlation", **px_arguments)
    figure = getattr(px, kind)(_plot_frame(df, max_points), title=title or None, **_px_values(px_arguments))
    updates = _trace_updates(kind, fixed, post_build, px_arguments.get("orientation"))
    if updates:
        figure.update_traces(**updates)
    if "barnorm" in post_build:
        figure.update_layout(barnorm=post_build["barnorm"])
    return figure


def plot_code(
    kind: str,
    arguments: dict[str, Any],
    title: str | None = None,
    max_points: int | None = DEFAULT_MAX_POINTS,
) -> str:
    """Plotly Express code equivalent to ``build_figure``."""
    px_arguments, fixed, post_build = _split_arguments(_clean_arguments(kind, arguments))
    lines = ["import plotly.express as px", ""]
    if kind == "correlation_heatmap":
        method = px_arguments.pop("method", "pearson")
        extra = f", {_px_code(px_arguments)}" if px_arguments else ""
        lines.append(
            f"fig = px.imshow(df.corr(numeric_only=True, method={method!r}), text_auto='.2f', "
            f"aspect='auto', title={title or 'Correlation'!r}{extra})"
        )
    else:
        data_variable = "df"
        if max_points:
            lines.append(f"plot_data = df.sample(min(len(df), {max_points}), random_state=0)  # keep the plot responsive")
            data_variable = "plot_data"
        keywords = _px_code({**px_arguments, **({"title": title} if title else {})})
        lines.append(f"fig = px.{kind}({data_variable}{', ' + keywords if keywords else ''})")
        updates = _trace_updates(kind, fixed, post_build, px_arguments.get("orientation"))
        if updates:
            lines.append(f"fig.update_traces({', '.join(f'{key}={value!r}' for key, value in updates.items())})")
        if "barnorm" in post_build:
            lines.append(f"fig.update_layout(barnorm={post_build['barnorm']!r})")
    lines.append("fig.show()")
    return "\n".join(lines)
