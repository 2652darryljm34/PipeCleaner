import numpy as np
import pandas as pd
import pytest

from data_cleaner.core import analysis
from data_cleaner.core.dataset import Dataset
from data_cleaner.core.loaders import load_data
from data_cleaner.core.workspace import Workspace


@pytest.fixture
def sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["N", "N", "S", "S", "S", None],
            "product": ["a", "b", "a", "a", "b", "a"],
            "units": [1, 2, 3, 4, 5, np.nan],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


def run_code(code: str, df: pd.DataFrame, result_name: str) -> pd.DataFrame:
    namespace = {"df": df, "pd": pd, "np": np}
    exec(code, namespace)  # noqa: S102 - testing our own generated code
    return namespace[result_name]


def test_column_overview_counts_distinct_and_nulls(sales):
    result = analysis.column_overview(sales)
    table = result.table
    assert table.loc["region", ["non_null", "nulls", "distinct"]].tolist() == [5, 1, 2]
    assert table.loc["units", "nulls"] == 1 and table.loc["price", "distinct"] == 6
    assert table.loc["product", "most_common"] == "a"
    pd.testing.assert_frame_equal(run_code(result.code, sales, "overview"), table)


def test_numeric_statistics_match_plain_pandas(sales):
    result = analysis.numeric_statistics(sales)
    table = result.table
    assert list(table.index) == ["units", "price"] and list(table.columns) == analysis.NUMERIC_STATISTICS
    assert table.loc["price", "sum"] == 210.0 and table.loc["price", "mean"] == 35.0
    assert table.loc["units", "count"] == 5  # the null is not counted
    assert table.loc["price", "std"] == pytest.approx(sales["price"].std())
    assert table.loc["price", "q1"] == sales["price"].quantile(0.25)
    pd.testing.assert_frame_equal(run_code(result.code, sales, "statistics"), table)


def test_value_counts_with_percent_nulls_and_limit(sales):
    result = analysis.value_counts(sales, "region")
    assert result.table["value"].tolist() == ["S", "N"] and result.table["count"].tolist() == [3, 2]
    assert result.table["percent"].tolist() == [60.0, 40.0]
    pd.testing.assert_frame_equal(run_code(result.code, sales, "counts"), result.table)

    with_nulls = analysis.value_counts(sales, "region", include_nulls=True, limit=2)
    assert len(with_nulls.table) == 2 and with_nulls.table["count"].sum() == 5
    pd.testing.assert_frame_equal(run_code(with_nulls.code, sales, "counts"), with_nulls.table)


def test_grouped_summary_does_not_modify_the_table_and_code_matches(sales):
    before = sales.copy()
    result = analysis.grouped_summary(sales, ["region"], {"price": ["sum", "mean"]}, include_row_count=True)
    assert result.table.columns.tolist() == ["region", "row_count", "price_sum", "price_mean"]
    assert result.table.set_index("region").loc["S", ["price_sum", "row_count"]].tolist() == [120.0, 3]
    pd.testing.assert_frame_equal(sales, before)

    namespace = {"df": sales, "pd": pd}
    exec(result.code, namespace)  # noqa: S102
    pd.testing.assert_frame_equal(namespace["summary"], result.table)
    pd.testing.assert_frame_equal(namespace["df"], before)  # the wrapper leaves df alone


def test_grouped_summary_row_count_only_and_validation(sales):
    only_counts = analysis.grouped_summary(sales, ["product"], {}, include_row_count=True)
    assert only_counts.table.columns.tolist() == ["product", "row_count"]
    assert only_counts.table["row_count"].tolist() == [4, 2]
    with pytest.raises(ValueError):
        analysis.grouped_summary(sales, ["product"], {}, include_row_count=False)


def test_pivot_summary(sales):
    result = analysis.pivot_summary(sales, ["product"], ["region"], ["price"], "sum")
    assert result.table.loc[result.table["product"] == "a", "price_S"].item() == 70.0
    namespace = {"df": sales, "pd": pd}
    exec(result.code, namespace)  # noqa: S102
    pd.testing.assert_frame_equal(namespace["summary"], result.table)


def test_analysis_only_operations_are_flagged():
    from data_cleaner.core.operations import OPERATIONS

    assert {name for name, op in OPERATIONS.items() if op.analysis_only} == {"group_by", "pivot"}


def test_saved_analysis_becomes_a_new_dataset_with_a_working_script(tmp_path, sales):
    path = tmp_path / "sales.csv"
    sales.to_csv(path, index=False)
    workspace = Workspace()
    source = workspace.add_loaded(load_data(path), "sales")
    source.apply("drop_nulls", subset=["region"])

    result = analysis.grouped_summary(source.current, ["region"], {"price": ["sum"]})
    saved = workspace.save_analysis("sales", "by_region", result)

    assert workspace.active_name == "by_region" and len(source.steps) == 1  # source untouched
    pd.testing.assert_frame_equal(saved.current, result.table)
    namespace: dict = {}
    exec(saved.export_script(), namespace)  # noqa: S102
    pd.testing.assert_frame_equal(namespace["build_by_region"](), saved.current)


def test_dataset_apply_still_supports_group_by_for_saved_pipelines(sales):
    dataset = Dataset("s", sales)
    dataset.apply("group_by", by=["product"], aggregations={"price": ["sum"]})
    assert dataset.current["price_sum"].tolist() == [140.0, 70.0]
