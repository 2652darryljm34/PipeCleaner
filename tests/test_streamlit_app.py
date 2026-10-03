"""Headless smoke tests of the Streamlit app using Streamlit's AppTest."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from data_cleaner.core.workspace import Workspace

APP_PATH = str(Path(__file__).parent.parent / "data_cleaner" / "streamlit_app.py")
SECTION_NAMES = ["Data", "Recommendations", "Clean", "Combine", "Quarantine", "Plot", "Code"]


@pytest.fixture
def app() -> AppTest:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "Customer Name": ["a ", "b", "B", "c"] * 25,
            "amount": [str(v) for v in rng.integers(0, 100, 100)],
            "region": ["N", "S"] * 50,
            "spend": rng.normal(100, 10, 100),
        }
    )
    workspace = Workspace()
    workspace.add_dataframe("sales", pd.concat([frame, frame.iloc[:3]], ignore_index=True))
    workspace.add_dataframe("other", frame.head(10))
    workspace.select("sales")  # adding a dataset activates it; the tests work on "sales"
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.session_state["workspace"] = workspace
    return app.run()


SECTION_VALUES = {
    "Data": ":material/table: Data",
    "Recommendations": ":material/lightbulb: Recommendations",
    "Clean": ":material/cleaning_services: Clean",
    "Combine": ":material/merge: Combine",
    "Quarantine": ":material/inventory_2: Quarantine",
    "Plot": ":material/bar_chart: Plot",
    "Code": ":material/code: Code",
}


def open_section(app: AppTest, name: str) -> AppTest:
    """Switch sections. AppTest lists options without icon markup but needs the full value."""
    control = next(widget for widget in app.main if getattr(widget, "key", None) == "section")
    control.set_value(SECTION_VALUES[name])
    app = app.run()
    current = next(widget for widget in app.main if getattr(widget, "key", None) == "section")
    assert current.value == SECTION_VALUES[name], "section did not switch"
    return app


def test_empty_app_shows_upload_prompt():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception
    assert "Upload a file" in app.info[0].value


@pytest.mark.parametrize("section", SECTION_NAMES)
def test_every_section_renders_without_errors(app, section):
    app = open_section(app, section)
    assert not app.exception, [e.value for e in app.exception]


def test_applying_a_recommendation_adds_a_step_and_undo_reverts(app):
    app = open_section(app, "Recommendations")
    next(button for button in app.button if button.key == "rec_apply_0").click()
    app.run()
    dataset = app.session_state["workspace"].active
    assert len(dataset.steps) == 1 and not app.exception
    next(button for button in app.sidebar.button if button.label == "Undo").click()
    app.run()
    assert len(dataset.steps) == 0


def test_clean_section_applies_drop_duplicates(app):
    app = open_section(app, "Clean")
    app.selectbox(key="clean_operation").select("drop_duplicates")
    app.run()
    assert any("drop_duplicates" in code.value for code in app.code)
    next(button for button in app.button if button.label == "Apply step").click()
    app.run()
    dataset = app.session_state["workspace"].active
    assert [step.operation for step in dataset.steps] == ["drop_duplicates"]
    assert len(dataset.removed_rows()) == 3


def test_plot_section_draws_a_chart(app):
    app = open_section(app, "Plot")
    app.selectbox(key="plot|scatter|x").select("spend")
    app.selectbox(key="plot|scatter|y").select("spend")
    app.run()
    assert not app.exception
    assert any("px.scatter" in code.value for code in app.code)


@pytest.mark.parametrize(
    "operation_name",
    [
        "rename_columns", "drop_duplicates", "drop_nulls", "filter_rows", "convert_dtype",
        "fill_nulls", "replace_values", "clean_text", "create_bins", "add_column",
        "drop_columns", "sort_values", "group_by", "pivot", "stack", "melt",
    ],
)  # fmt: skip
def test_every_clean_form_renders(app, operation_name):
    app = open_section(app, "Clean")
    app.selectbox(key="clean_operation").select(operation_name)
    app.run()
    assert not app.exception, [e.value for e in app.exception]


def test_filter_form_previews_and_applies(app):
    app = open_section(app, "Clean")
    app.selectbox(key="clean_operation").select("filter_rows")
    app.run()
    prefix = "sales|0|filter_rows"
    app.selectbox(key=f"{prefix}_col0").select("spend")
    app.selectbox(key=f"{prefix}_op0").select(">")
    app.text_input(key=f"{prefix}_val0").input("100")
    app.run()
    assert not app.exception and any("keep_row" in code.value for code in app.code)
    next(button for button in app.button if button.label == "Apply step").click()
    app.run()
    dataset = app.session_state["workspace"].active
    assert (dataset.current["spend"] > 100).all() and len(dataset.removed_rows()) > 0


@pytest.mark.parametrize(
    "kind",
    [
        "scatter", "line", "bar", "histogram", "box", "violin", "strip", "area", "pie", "density_heatmap",
        "density_contour", "ecdf", "scatter_3d", "scatter_matrix", "treemap", "sunburst", "funnel", "correlation_heatmap",
    ],
)  # fmt: skip
def test_every_plot_kind_renders_all_its_widgets(app, kind):
    app = open_section(app, "Plot")
    app.selectbox(key="plot_kind").select(kind)
    app.run()
    assert not app.exception, [e.value for e in app.exception]


def test_plot_widgets_match_argument_types(app):
    app = open_section(app, "Plot")
    app.selectbox(key="plot_kind").select("histogram")
    app.run()
    assert app.number_input(key="plot|histogram|nbins").value is None  # empty = automatic
    assert app.number_input(key="plot|histogram|bin_size") is not None
    assert app.checkbox(key="plot|histogram|cumulative").value is False
    assert app.selectbox(key="plot|histogram|histnorm") is not None

    app.selectbox(key="plot_kind").select("scatter")
    app.run()
    assert app.text_input(key="plot|scatter|color_fixed") is not None
    assert app.number_input(key="plot|scatter|size_fixed") is not None
    assert app.selectbox(key="plot|scatter|symbol_fixed") is not None


def test_fixed_color_and_size_from_the_ui_reach_the_chart_code(app):
    app = open_section(app, "Plot")
    app.selectbox(key="plot|scatter|x").select("spend")
    app.selectbox(key="plot|scatter|y").select("spend")
    app.text_input(key="plot|scatter|color_fixed").input("crimson")
    app.run()
    app.number_input(key="plot|scatter|size_fixed").set_value(14)
    app.run()
    code = "\n".join(block.value for block in app.code)
    assert not app.exception and "marker_color='crimson'" in code and "marker_size=14" in code

    # Choosing a column disables the fixed value and wins.
    app.selectbox(key="plot|scatter|color").select("region")
    app.run()
    assert app.text_input(key="plot|scatter|color_fixed").disabled
    assert "marker_color" not in "\n".join(block.value for block in app.code)
