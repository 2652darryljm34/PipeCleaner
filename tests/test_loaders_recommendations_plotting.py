import numpy as np
import pandas as pd
import pytest

from data_cleaner.core.dataset import Dataset
from data_cleaner.core.loaders import load_data
from data_cleaner.core.plotting import PLOT_KINDS, build_figure, plot_code
from data_cleaner.core.recommendations import recommend


@pytest.mark.parametrize("extension", ["csv", "tsv", "json", "xlsx", "parquet"])
def test_load_each_supported_format(tmp_path, extension):
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / f"data.{extension}"
    writers = {
        "csv": lambda: frame.to_csv(path, index=False),
        "tsv": lambda: frame.to_csv(path, index=False, sep="\t"),
        "json": lambda: frame.to_json(path),
        "xlsx": lambda: frame.to_excel(path, index=False),
        "parquet": lambda: frame.to_parquet(path),
    }
    writers[extension]()
    loaded = load_data(path)
    assert loaded.dataframe["a"].tolist() == [1, 2]
    assert loaded.suggested_name == "data"
    namespace = {"pd": pd}
    exec(loaded.origin_code, namespace)  # noqa: S102
    assert namespace["df"]["a"].tolist() == [1, 2]


def test_load_uploaded_file_object(tmp_path):
    path = tmp_path / "up.csv"
    path.write_text("a,b\n1,2\n")
    with open(path, "rb") as handle:
        loaded = load_data(handle, filename="up.csv")
    assert loaded.dataframe.shape == (1, 2)
    assert "Uploaded file" in loaded.origin_code


def test_recommendations_find_planted_problems():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "Customer Name": ["a ", "b", "B", "c"] * 25,
            "amount": [str(v) for v in rng.integers(0, 100, 100)],
            "signup": pd.date_range("2024-01-01", periods=100).astype(str),
            "mostly_empty": [np.nan] * 90 + [1.0] * 10,
            "constant": 1,
            "value": np.append(rng.normal(50, 5, 99), 5000),
        }
    )
    frame = pd.concat([frame, frame.iloc[:3]], ignore_index=True)
    titles = " | ".join(r.title for r in recommend(frame))
    for expected in [
        "duplicate",
        "Trim whitespace",
        "'amount' to int",
        "'signup' to datetime",
        "mostly-empty column 'mostly_empty'",
        "constant column",
        "snake_case",
        "outlier",
    ]:
        assert expected in titles, expected


def test_every_recommendation_applies_cleanly():
    frame = pd.DataFrame({"Some Col": [" a", "A", None, "a"] * 30, "n": ["1", "2", "3", "x"] * 30})
    dataset = Dataset("t", frame)
    recommendations = recommend(frame)
    assert recommendations
    for recommendation in recommendations:
        dataset.preview(recommendation.operation, **recommendation.params)
        recommendation.code()


PLOT_ARGUMENTS = {
    "scatter": {"x": "x", "y": "y", "color": "g"},
    "line": {"x": "x", "y": "y"},
    "bar": {"x": "g", "y": "y"},
    "histogram": {"x": "x", "nbins": 10},
    "box": {"y": "y", "x": "g"},
    "violin": {"y": "y"},
    "strip": {"y": "y"},
    "area": {"x": "x", "y": "y"},
    "pie": {"names": "g"},
    "density_heatmap": {"x": "x", "y": "y"},
    "density_contour": {"x": "x", "y": "y"},
    "ecdf": {"x": "x"},
    "scatter_3d": {"x": "x", "y": "y", "z": "z"},
    "scatter_matrix": {"dimensions": ["x", "y", "z"]},
    "treemap": {"path": ["g", "h"]},
    "sunburst": {"path": ["g", "h"]},
    "funnel": {"x": "x", "y": "g"},
    "correlation_heatmap": {},
}


def test_plot_arguments_cover_every_kind():
    assert set(PLOT_ARGUMENTS) == set(PLOT_KINDS)


@pytest.mark.parametrize("kind", sorted(PLOT_ARGUMENTS))
def test_every_plot_kind_builds_and_generates_code(kind):
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=60),
            "y": rng.normal(size=60),
            "z": rng.normal(size=60),
            "g": list("abc") * 20,
            "h": list("xy") * 30,
        }
    )
    arguments = PLOT_ARGUMENTS[kind]
    assert build_figure(frame, kind, arguments, title="t").data
    namespace = {"df": frame}
    exec(plot_code(kind, arguments, "t").replace("fig.show()", ""), namespace)  # noqa: S102
    assert namespace["fig"].data


def test_missing_required_plot_argument_is_reported():
    with pytest.raises(ValueError, match="need: x, y"):
        build_figure(pd.DataFrame({"a": [1]}), "scatter", {})


# ---------------------------------------------------------------- argument types

import inspect  # noqa: E402

import plotly.express as px  # noqa: E402

from data_cleaner.core.plotting import _POST_BUILD_ARGUMENTS, fixed_key, resolve_fixed_arguments  # noqa: E402

STYLE_FRAME = pd.DataFrame(
    {"x": np.arange(40.0), "y": np.arange(40.0) ** 2, "g": list("ab") * 20, "w": np.arange(40.0) + 1}
)


def run_plot_code(kind, arguments, title=None):
    namespace = {"df": STYLE_FRAME}
    exec(plot_code(kind, arguments, title).replace("fig.show()", ""), namespace)  # noqa: S102
    return namespace["fig"]


def test_every_declared_argument_exists_in_plotly_express():
    """Catches typos in the argument table before users hit them."""
    for kind, specs in PLOT_KINDS.items():
        if kind == "correlation_heatmap":
            continue
        accepted = set(inspect.signature(getattr(px, kind)).parameters) | _POST_BUILD_ARGUMENTS
        unknown = [spec.name for spec in specs if spec.name not in accepted]
        assert not unknown, f"{kind}: {unknown}"


def test_fixed_color_size_and_symbol_apply_to_all_points():
    arguments = {"x": "x", "y": "y", fixed_key("color"): "red", fixed_key("size"): 15, fixed_key("symbol"): "diamond"}
    for figure in (build_figure(STYLE_FRAME, "scatter", arguments), run_plot_code("scatter", arguments)):
        marker = figure.data[0].marker
        assert (marker.color, marker.size, marker.symbol) == ("red", 15, "diamond")


def test_column_mapping_wins_over_fixed_value():
    figure = build_figure(STYLE_FRAME, "scatter", {"x": "x", "y": "y", "color": "g", fixed_key("color"): "red"})
    assert len(figure.data) == 2  # one trace per category, not a single red trace


def test_fixed_color_on_line_style_kinds():
    figure = build_figure(STYLE_FRAME, "line", {"x": "x", "y": "y", fixed_key("color"): "green"})
    assert figure.data[0].line.color == "green"


def test_cli_style_values_are_resolved_into_fixed_values():
    resolved = resolve_fixed_arguments(STYLE_FRAME, "scatter", {"x": "x", "y": "y", "color": "g", "size": 12, "symbol": "star"})
    assert resolved["color"] == "g" and resolved[fixed_key("size")] == 12 and resolved[fixed_key("symbol")] == "star"
    assert resolve_fixed_arguments(STYLE_FRAME, "scatter", {"color": "red"})[fixed_key("color")] == "red"


def test_numeric_and_style_arguments_reach_the_figure_and_the_code():
    for kind, arguments in [
        ("histogram", {"x": "x", "nbins": 5, "bin_size": 10.0, "opacity": 0.5, "cumulative": True, "histnorm": "percent"}),
        ("pie", {"names": "g", "hole": 0.4, "color_discrete_sequence": "Set2"}),
        ("scatter", {"x": "x", "y": "y", "log_y": True, "marginal_x": "box", "template": "plotly_dark"}),
        ("bar", {"x": "g", "y": "y", "barmode": "group", "text_auto": True, "orientation": "v"}),
        ("correlation_heatmap", {"method": "spearman", "color_continuous_scale": "Viridis"}),
    ]:
        assert build_figure(STYLE_FRAME, kind, arguments).data, kind
        assert run_plot_code(kind, arguments).data, kind
    histogram = build_figure(STYLE_FRAME, "histogram", {"x": "x", "bin_size": 10.0})
    assert histogram.data[0].xbins.size == 10.0
    assert build_figure(STYLE_FRAME, "pie", {"names": "g", "hole": 0.4}).data[0].hole == 0.4
    stacked = {"x": "g", "y": "y", "color": "g", "barnorm": "percent"}
    assert build_figure(STYLE_FRAME, "bar", stacked).layout.barnorm == "percent"
    assert run_plot_code("bar", stacked).layout.barnorm == "percent"


def test_false_toggles_are_left_out_of_the_code():
    code = plot_code("scatter", {"x": "x", "y": "y", "log_x": False, "opacity": 0.5})
    assert "log_x" not in code and "opacity=0.5" in code
