"""A workspace holds several named datasets and can merge/concatenate them."""

from __future__ import annotations

import textwrap

import pandas as pd

from .analysis import AnalysisResult
from .dataset import Dataset
from .loaders import LoadedData
from .operations import OperationError


def _loader_function(function_name: str, dataset: Dataset) -> str:
    """Nested function that rebuilds ``dataset`` as it is right now (a frozen snapshot)."""
    body = textwrap.indent(dataset.body_code() + "\n\nreturn df", "    ")
    return f"def {function_name}() -> pd.DataFrame:\n{body}\n"


class Workspace:
    """Named datasets plus the notion of an active one.

    Merged/concatenated datasets are snapshots: they start from the combined result
    and do not change if their source datasets are edited afterwards.
    """

    def __init__(self) -> None:
        self.datasets: dict[str, Dataset] = {}
        self.active_name: str | None = None

    @property
    def active(self) -> Dataset | None:
        return self.datasets.get(self.active_name) if self.active_name else None

    def add_loaded(self, loaded: LoadedData, name: str | None = None) -> Dataset:
        return self._add(Dataset(self._unique_name(name or loaded.suggested_name), loaded.dataframe, loaded.origin_code))

    def add_dataframe(self, name: str, dataframe: pd.DataFrame, origin_code: str = "") -> Dataset:
        return self._add(Dataset(self._unique_name(name), dataframe, origin_code))

    def remove(self, name: str) -> None:
        self.datasets.pop(name, None)
        if self.active_name == name:
            self.active_name = next(iter(self.datasets), None)

    def select(self, name: str) -> None:
        if name not in self.datasets:
            raise OperationError(f"No dataset named '{name}'.")
        self.active_name = name

    # ------------------------------------------------------------------- combining

    def merge(
        self,
        left_name: str,
        right_name: str,
        new_name: str,
        how: str = "inner",
        left_on: list[str] | None = None,
        right_on: list[str] | None = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> Dataset:
        """SQL-style join of two datasets into a new dataset."""
        left, right = self._get(left_name), self._get(right_name)
        keys = {"how": how, "left_on": left_on, "right_on": right_on, "suffixes": tuple(suffixes)}
        try:
            merged = left.current.merge(right.current, **keys)
        except (KeyError, ValueError) as error:
            raise OperationError(f"Merge failed: {error}") from error
        arguments = ", ".join(f"{key}={value!r}" for key, value in keys.items())
        origin_code = (
            f"{_loader_function('load_left', left)}\n"
            f"{_loader_function('load_right', right)}\n"
            f"df = load_left().merge(load_right(), {arguments})"
        )
        return self._add(Dataset(self._unique_name(new_name), merged, origin_code))

    def concat(
        self,
        names: list[str],
        new_name: str,
        axis: int = 0,
        join: str = "outer",
        ignore_index: bool = True,
    ) -> Dataset:
        """Stack datasets on top of each other (axis=0) or side by side (axis=1)."""
        if len(names) < 2:
            raise OperationError("Choose at least two datasets to concatenate.")
        sources = [self._get(name) for name in names]
        keys = {"axis": axis, "join": join, "ignore_index": ignore_index}
        try:
            combined = pd.concat([source.current for source in sources], **keys)
        except (KeyError, ValueError) as error:
            raise OperationError(f"Concat failed: {error}") from error
        loaders = [f"load_dataset_{number}" for number in range(1, len(sources) + 1)]
        functions = "\n".join(_loader_function(fn, ds) for fn, ds in zip(loaders, sources))
        calls = ", ".join(f"{fn}()" for fn in loaders)
        arguments = ", ".join(f"{key}={value!r}" for key, value in keys.items())
        origin_code = f"{functions}\ndf = pd.concat([{calls}], {arguments})"
        return self._add(Dataset(self._unique_name(new_name), combined, origin_code))

    def save_analysis(self, source_name: str, new_name: str, result: AnalysisResult) -> Dataset:
        """Keep an analysis result (e.g. a group-by summary) as its own dataset.

        The source dataset is not changed. The new dataset starts from the result table, and
        its exported script rebuilds it from the source as it is right now.
        """
        source = self._get(source_name)
        origin_code = (
            f"{_loader_function('load_source', source)}\n"
            f"df = load_source()\n{result.code}\ndf = summary"
        )
        return self._add(Dataset(self._unique_name(new_name), result.table, origin_code))

    # ------------------------------------------------------------------- internals

    def _get(self, name: str) -> Dataset:
        if name not in self.datasets:
            raise OperationError(f"No dataset named '{name}'.")
        return self.datasets[name]

    def _add(self, dataset: Dataset) -> Dataset:
        self.datasets[dataset.name] = dataset
        self.active_name = dataset.name
        return dataset

    def _unique_name(self, name: str) -> str:
        candidate, counter = name, 2
        while candidate in self.datasets:
            candidate, counter = f"{name} ({counter})", counter + 1
        return candidate
