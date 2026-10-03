import pandas as pd
import pytest

from data_cleaner.core.dataset import Dataset, StepReplayError
from data_cleaner.core.loaders import load_data
from data_cleaner.core.operations import OperationError
from data_cleaner.core.workspace import Workspace


@pytest.fixture
def csv_path(tmp_path, messy_frame):
    path = tmp_path / "messy.csv"
    messy_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def dataset(csv_path) -> Dataset:
    loaded = load_data(csv_path)
    return Dataset("messy", loaded.dataframe, loaded.origin_code)


def run_exported_script(script: str, function_name: str) -> pd.DataFrame:
    namespace: dict = {}
    exec(script, namespace)  # noqa: S102 - testing our own generated script
    return namespace[function_name]()


def test_original_is_preserved(dataset):
    before = dataset.original.copy()
    dataset.apply("drop_nulls", subset=["age"])
    dataset.apply("drop_columns", columns=["city"])
    pd.testing.assert_frame_equal(dataset.original, before)


def test_undo_redo_and_redo_tail_is_discarded(dataset):
    dataset.apply("drop_nulls", subset=["age"])
    dataset.apply("drop_columns", columns=["city"])
    dataset.undo()
    assert "city" in dataset.current.columns and dataset.can_redo
    dataset.redo()
    assert "city" not in dataset.current.columns
    dataset.undo()
    dataset.apply("sort_values", by=["age"])
    assert not dataset.can_redo and len(dataset.steps) == 2


def test_quarantine_collects_removed_rows_by_step(dataset):
    dataset.apply("drop_duplicates", subset=["city"])
    dataset.apply("drop_nulls", subset=["age"])
    removed = dataset.removed_rows()
    assert removed["_step"].tolist().count(1) == 3
    assert list(removed.columns[:4]) == ["_step", "_operation", "_reason", "_row"]
    dataset.undo()
    assert set(dataset.removed_rows()["_step"]) == {1}


def test_restore_rows_puts_rows_back_and_clears_quarantine(dataset):
    dataset.apply("drop_nulls", subset=["age"])
    assert len(dataset.removed_rows()) == 2
    dataset.restore_rows([0])
    assert len(dataset.removed_rows()) == 1
    assert len(dataset.current) == 5
    rebuilt = run_exported_script(dataset.export_script(), "build_messy")
    pd.testing.assert_frame_equal(rebuilt, dataset.current, check_dtype=False)


def test_failed_replay_changes_nothing(dataset):
    dataset.apply("rename_columns", mapping={"city": "town"})
    dataset.apply("sort_values", by=["town"])
    with pytest.raises(StepReplayError) as error:
        dataset.remove_step(0)  # sorting by 'town' is impossible without the rename
    assert error.value.step_number == 1  # position in the proposed history
    assert len(dataset.steps) == 2 and "town" in dataset.current.columns


def test_edit_step_replays_downstream_steps(dataset):
    dataset.apply("drop_nulls", subset=["age"])
    dataset.apply("sort_values", by=["age"])
    dataset.edit_step(0, {"subset": ["Name"], "how": "any"})
    assert dataset.current["Name"].notna().all()
    assert dataset.current["age"].dropna().is_monotonic_increasing


def test_editor_changes_become_steps(dataset):
    dataset.apply_editor_changes(
        edited_rows={0: {"age": 26.0}}, added_rows=[{"Name": "Dee", "age": 1.0}], deleted_rows=[1]
    )
    assert [step.operation for step in dataset.steps] == ["set_cells", "drop_rows", "add_rows"]
    assert 26.0 in dataset.current["age"].tolist() and "Dee" in dataset.current["Name"].tolist()
    assert len(dataset.removed_rows()) == 1 and len(dataset.changed_cells()) == 1


def test_exported_script_reproduces_current_table(dataset):
    dataset.apply("clean_text", columns=["Name"], strip=True, case="lower")
    dataset.apply("convert_dtype", column="score", dtype="float")
    dataset.apply("group_by", by=["city"], aggregations={"score": ["sum"]})
    rebuilt = run_exported_script(dataset.export_script(), "build_messy")
    pd.testing.assert_frame_equal(rebuilt, dataset.current)


def test_steps_roundtrip_through_json(dataset, csv_path):
    dataset.apply("drop_nulls", subset=["age"])
    dataset.apply("fill_nulls", columns=["Name"], strategy="value", value="unknown")
    other = Dataset("copy", load_data(csv_path).dataframe)
    other.load_steps_json(dataset.steps_to_json())
    pd.testing.assert_frame_equal(other.current, dataset.current)


def test_workspace_merge_and_concat_scripts_run(csv_path, tmp_path):
    workspace = Workspace()
    workspace.add_loaded(load_data(csv_path), "people")
    lookup = pd.DataFrame({"city": ["NY", "LA"], "state": ["New York", "California"]})
    lookup_path = tmp_path / "states.csv"
    lookup.to_csv(lookup_path, index=False)
    workspace.add_loaded(load_data(lookup_path), "states")

    merged = workspace.merge("people", "states", "joined", how="inner", left_on=["city"], right_on=["city"])
    assert "state" in merged.current.columns and len(merged.current) == 4
    merged.apply("drop_columns", columns=["score"])
    rebuilt = run_exported_script(merged.export_script(), "build_joined")
    pd.testing.assert_frame_equal(rebuilt, merged.current)

    stacked = workspace.concat(["people", "people"], "twice")
    assert len(stacked.current) == 12
    pd.testing.assert_frame_equal(run_exported_script(stacked.export_script(), "build_twice"), stacked.current)

    assert workspace.active_name == "twice"
    with pytest.raises(OperationError):
        workspace.concat(["people"], "nope")


def test_duplicate_dataset_names_get_a_suffix(csv_path):
    workspace = Workspace()
    first = workspace.add_loaded(load_data(csv_path))
    second = workspace.add_loaded(load_data(csv_path))
    assert (first.name, second.name) == ("messy", "messy (2)")
