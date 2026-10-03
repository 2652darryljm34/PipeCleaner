"""Every operation's generated code must produce the same table as ``apply``."""

import numpy as np
import pandas as pd
import pytest

from data_cleaner.core.operations import OPERATIONS, OperationError

CASES = {
    "rename_columns": {"mapping": {"Name": "name"}},
    "drop_duplicates": {"subset": ["city"], "keep": "first"},
    "drop_nulls": {"subset": ["age"], "how": "any"},
    "filter_rows": {
        "conditions": [
            {"column": "age", "operator": ">", "value": "26"},
            {"column": "city", "operator": "is in", "value": "NY, LA"},
        ],
        "combine": "and",
    },
    "convert_dtype": {"column": "score", "dtype": "int"},
    "fill_nulls": {"columns": ["age"], "strategy": "value", "value": "0"},
    "replace_values": {"columns": ["city"], "to_replace": ["NY"], "value": "New York"},
    "clean_text": {"columns": ["Name"], "strip": True, "case": "lower"},
    "create_bins": {"column": "age", "method": "custom", "bins": "0,30,50", "labels": "young,old"},
    "add_column": {"name": "age2", "mode": "expression", "expression": "age * 2"},
    "drop_columns": {"columns": ["city"]},
    "sort_values": {"by": ["age"], "ascending": False},
    "group_by": {"by": ["city"], "aggregations": {"age": ["mean", "count"]}},
    "pivot": {"index": ["Name"], "columns": ["city"], "values": ["age"], "aggfunc": "sum"},
    "stack": {"id_columns": ["Name"]},
    "melt": {"id_vars": ["Name"], "value_vars": ["age", "score"]},
    "set_cells": {"edits": [{"row": 0, "column": "age", "value": 26.0}]},
    "drop_rows": {"labels": [0, 1]},
    "add_rows": {"records": [{"Name": "Dee", "age": 50.0, "score": "9", "city": "TX"}]},
}


@pytest.mark.parametrize("operation_name", sorted(CASES))
def test_generated_code_matches_apply(operation_name, messy_frame):
    operation = OPERATIONS[operation_name]
    params = operation.normalize(messy_frame, CASES[operation_name])

    expected = operation.apply(messy_frame, **params).data

    namespace = {"df": messy_frame.copy(), "pd": pd, "np": np}
    exec(operation.to_code(**params), namespace)  # noqa: S102 - testing our own generated code
    pd.testing.assert_frame_equal(namespace["df"], expected)


def test_operations_do_not_mutate_input(messy_frame):
    snapshot = messy_frame.copy()
    for name, params in CASES.items():
        operation = OPERATIONS[name]
        operation.apply(messy_frame, **operation.normalize(messy_frame, params))
    pd.testing.assert_frame_equal(messy_frame, snapshot)


def test_every_operation_is_covered_or_special():
    assert set(OPERATIONS) - set(CASES) == {"restore_rows"}  # tested in test_dataset


def test_drop_duplicates_quarantines_with_reason(messy_frame):
    result = OPERATIONS["drop_duplicates"].apply(messy_frame, subset=["city"])
    assert len(result.removed_rows) == 3
    assert result.removed_rows["_reason"].str.contains("duplicate").all()


def test_drop_nulls_names_the_null_columns(messy_frame):
    result = OPERATIONS["drop_nulls"].apply(messy_frame, subset=["age", "Name"])
    assert set(result.removed_rows["_reason"]) == {"null in: age", "null in: Name"}


def test_failed_conversions_are_recorded_as_changed_cells(messy_frame):
    result = OPERATIONS["convert_dtype"].apply(messy_frame, column="score", dtype="int")
    assert result.changed_cells["before"].tolist() == ["x"]
    assert result.data["score"].isna().sum() == 1


def test_fill_nulls_records_only_cells_that_changed(messy_frame):
    result = OPERATIONS["fill_nulls"].apply(
        messy_frame, columns=["age"], strategy="median"
    )
    assert len(result.changed_cells) == 2
    assert result.changed_cells["after"].tolist() == [31.0, 31.0]


def test_filter_keeps_nulls_out_and_quarantines_them(messy_frame):
    operation = OPERATIONS["filter_rows"]
    params = operation.normalize(
        messy_frame, {"conditions": [{"column": "age", "operator": ">", "value": "30"}]}
    )
    result = operation.apply(messy_frame, **params)
    assert result.data["age"].min() > 30
    assert len(result.removed_rows) == 3  # 25, and two NaN


def test_bad_inputs_raise_operation_error(messy_frame):
    with pytest.raises(OperationError):
        OPERATIONS["drop_columns"].apply(messy_frame, columns=["nope"])
    with pytest.raises(OperationError):
        OPERATIONS["add_column"].apply(messy_frame, name="x", expression="nope + 1")
    with pytest.raises(OperationError):
        OPERATIONS["fill_nulls"].apply(messy_frame, columns=["city"], strategy="mean")
