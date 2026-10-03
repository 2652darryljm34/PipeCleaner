"""A dataset: the untouched original plus an undoable, replayable list of cleaning steps."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .operations import OperationError, OperationResult, Params, get_operation
from .operations.manual_edits import rows_to_split_dict

REMOVED_META_COLUMNS = ["_step", "_operation", "_row", "_reason"]
CHANGED_META_COLUMNS = ["_step", "_operation", "row", "column", "before", "after", "reason"]


@dataclass(frozen=True)
class Step:
    """One recorded operation: its registry name and JSON-friendly parameters."""

    operation: str
    params: Params

    def label(self, max_length: int = 90) -> str:
        arguments = ", ".join(f"{key}={value!r}" for key, value in self.params.items())
        text = f"{self.operation}({arguments})"
        return text if len(text) <= max_length else text[: max_length - 3] + "..."

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "params": self.params}


class StepReplayError(OperationError):
    """A step failed while replaying history (e.g. after editing or removing an earlier step)."""

    def __init__(self, step_number: int, step: Step, cause: Exception):
        super().__init__(
            f"After this change, step {step_number} ({step.operation}) would fail: {cause}"
        )
        self.step_number = step_number


class Dataset:
    """Original data + steps. The current table is always ``original`` replayed through ``steps``.

    Steps live in a list with a ``cursor``: steps before the cursor are active, steps
    after it are available to redo. Applying a new step discards the redo tail.
    """

    def __init__(
        self, name: str, original: pd.DataFrame, origin_code: str = "", source_description: str = ""
    ):
        self.name = name
        self._original = original.copy()  # copy-on-write: callers can never alter it
        self.origin_code = origin_code or "df = pd.DataFrame()  # origin unknown"
        self.source_description = source_description
        self._steps: list[Step] = []
        self._results: list[OperationResult] = []  # _results[i] is the outcome of _steps[i]
        self._cursor = 0

    # ------------------------------------------------------------------ state access

    @property
    def original(self) -> pd.DataFrame:
        return self._original

    @property
    def current(self) -> pd.DataFrame:
        return self._results[self._cursor - 1].data if self._cursor else self._original

    @property
    def steps(self) -> list[Step]:
        """Active steps (those before the cursor)."""
        return self._steps[: self._cursor]

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._steps)

    # --------------------------------------------------------------------- mutations

    def preview(self, operation_name: str, **params: Any) -> OperationResult:
        """Run an operation without recording it (used to preview effects before applying)."""
        operation = get_operation(operation_name)
        return operation.apply(self.current, **operation.normalize(self.current, params))

    def apply(self, operation_name: str, **params: Any) -> OperationResult:
        """Run an operation on the current table and record it as a new step."""
        operation = get_operation(operation_name)
        normalized = operation.normalize(self.current, params)
        result = operation.apply(self.current, **normalized)
        del self._steps[self._cursor :], self._results[self._cursor :]  # drop the redo tail
        self._steps.append(Step(operation_name, normalized))
        self._results.append(result)
        self._cursor += 1
        return result

    def undo(self) -> None:
        if not self.can_undo:
            raise OperationError("Nothing to undo.")
        self._cursor -= 1

    def redo(self) -> None:
        if not self.can_redo:
            raise OperationError("Nothing to redo.")
        self._cursor += 1

    def reset(self) -> None:
        """Go back to the original data (steps stay available to redo)."""
        self._cursor = 0

    def remove_step(self, index: int) -> None:
        """Delete the active step at ``index`` (0-based) and replay the steps after it."""
        steps = self.steps
        self._replace_steps_from(index, steps[:index] + steps[index + 1 :])

    def edit_step(self, index: int, params: Params) -> None:
        """Change the parameters of an active step and replay the steps after it."""
        steps = self.steps
        steps[index] = Step(steps[index].operation, params)
        self._replace_steps_from(index, steps)

    def replace_steps(self, steps: list[Step]) -> None:
        """Replace the whole history (e.g. when loading a saved pipeline)."""
        self._replace_steps_from(0, steps)

    def _replace_steps_from(self, start: int, steps: list[Step]) -> None:
        """Replay ``steps[start:]`` on top of the unchanged earlier results, atomically.

        On failure nothing is modified and a StepReplayError names the failing step.
        """
        results = self._results[:start]
        frame = results[-1].data if results else self._original
        rebuilt_steps = list(steps[:start])
        for number, step in enumerate(steps[start:], start=start + 1):
            try:
                operation = get_operation(step.operation)
                params = operation.normalize(frame, step.params)
                result = operation.apply(frame, **params)
            except Exception as error:
                raise StepReplayError(number, step, error) from error
            rebuilt_steps.append(Step(step.operation, params))
            results.append(result)
            frame = result.data
        self._steps, self._results, self._cursor = rebuilt_steps, results, len(rebuilt_steps)

    # ------------------------------------------------------------- manual table edits

    def apply_editor_changes(
        self,
        edited_rows: dict[int, dict[str, Any]],
        added_rows: list[dict[str, Any]],
        deleted_rows: list[int],
    ) -> None:
        """Record changes made in an editable grid (positions refer to ``current``)."""
        row_labels = list(self.current.index)
        if edited_rows:
            edits = [
                {"row": row_labels[position], "column": column, "value": value}
                for position, changes in edited_rows.items()
                for column, value in changes.items()
            ]
            self.apply("set_cells", edits=edits)
        if deleted_rows:
            self.apply("drop_rows", labels=[row_labels[position] for position in deleted_rows])
        if added_rows:
            self.apply("add_rows", records=added_rows)

    def restore_rows(self, quarantine_positions: list[int]) -> None:
        """Move rows from the quarantine table (by position in ``removed_rows()``) back in."""
        selected = self.removed_rows().iloc[quarantine_positions]
        restored = selected.drop(columns=REMOVED_META_COLUMNS).set_axis(selected["_row"].tolist())
        sources = [[int(step), row] for step, row in zip(selected["_step"], selected["_row"].tolist())]
        self.apply("restore_rows", sources=sources, rows=rows_to_split_dict(restored))

    # -------------------------------------------------------------------- quarantine

    def removed_rows(self) -> pd.DataFrame:
        """All rows removed by active steps (minus restored ones), tagged by step and reason."""
        restored = {
            (source_step, source_row)
            for step in self.steps
            if step.operation == "restore_rows"
            for source_step, source_row in step.params["sources"]
        }
        frames = []
        for number, (step, result) in enumerate(zip(self.steps, self._results), start=1):
            if result.removed_rows is None or result.removed_rows.empty:
                continue
            frame = result.removed_rows.copy()
            frame.insert(0, "_row", frame.index)
            frame.insert(0, "_operation", step.operation)
            frame.insert(0, "_step", number)
            frames.append(frame.reset_index(drop=True))
        if not frames:
            return pd.DataFrame(columns=REMOVED_META_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        is_restored = [(s, r) in restored for s, r in zip(combined["_step"], combined["_row"])]
        combined = combined.loc[[not flag for flag in is_restored]].reset_index(drop=True)
        data_columns = [c for c in combined.columns if c not in REMOVED_META_COLUMNS]
        return combined[["_step", "_operation", "_reason", "_row", *data_columns]]

    def changed_cells(self) -> pd.DataFrame:
        """Every cell modified by active steps, with its value before and after."""
        frames = []
        for number, (step, result) in enumerate(zip(self.steps, self._results), start=1):
            if result.changed_cells is None or result.changed_cells.empty:
                continue
            frame = result.changed_cells.copy()
            frame.insert(0, "_operation", step.operation)
            frame.insert(0, "_step", number)
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=CHANGED_META_COLUMNS)
        return pd.concat(frames, ignore_index=True)[CHANGED_META_COLUMNS]

    # ------------------------------------------------------------ code / persistence

    def step_code(self, step_number: int) -> str:
        """pandas code for the active step with 1-based ``step_number``."""
        step = self.steps[step_number - 1]
        return get_operation(step.operation).to_code(**step.params)

    def body_code(self) -> str:
        """Code (reading/writing ``df``) that rebuilds the current table from its source."""
        lines = [self.origin_code]
        for number, step in enumerate(self.steps, start=1):
            lines.append(f"\n# Step {number}: {get_operation(step.operation).summary}")
            lines.append(self.step_code(number))
        return "\n".join(lines)

    def export_script(self) -> str:
        """A standalone Python script that reproduces the current table."""
        function_name = "build_" + re.sub(r"\W+", "_", self.name).strip("_").lower()
        body = textwrap.indent(self.body_code() + "\n\nreturn df", "    ")
        return (
            f'"""Generated by PipeCleaner: rebuilds the dataset {self.name!r}."""\n\n'
            "import numpy as np\nimport pandas as pd\n\n\n"
            f"def {function_name}() -> pd.DataFrame:\n{body}\n\n\n"
            f'if __name__ == "__main__":\n    print({function_name}().head())\n'
        )

    def steps_to_json(self) -> str:
        return json.dumps([step.to_dict() for step in self.steps], indent=2, default=str)

    def load_steps_json(self, text: str) -> None:
        """Replace the history with steps saved by ``steps_to_json``."""
        try:
            steps = [Step(item["operation"], item["params"]) for item in json.loads(text)]
        except (ValueError, KeyError, TypeError) as error:
            raise OperationError(f"Not a valid steps file: {error}") from error
        self.replace_steps(steps)
