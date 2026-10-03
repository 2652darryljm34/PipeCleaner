"""Interactive command-line REPL (and one-shot runner) for PipeCleaner.

Every cleaning operation is available as a command, e.g.::

    pipecleaner> load sales.csv
    pipecleaner> drop_nulls subset=price,qty
    pipecleaner> filter_rows conditions='[{"column":"price","operator":">","value":0}]'
    pipecleaner> quarantine
    pipecleaner> export clean.csv

After each operation the exact pandas code that ran is printed.
"""

from __future__ import annotations

import argparse
import cmd
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .core import analysis
from .core.dataset import Dataset
from .core.loaders import SUPPORTED_EXTENSIONS, load_data
from .core.operations import OPERATIONS, OperationError, parse_cli_value
from .core.operations.base import as_list
from .core.plotting import PLOT_KINDS, build_figure, plot_code, resolve_fixed_arguments
from .core.recommendations import apply_all_recommendations, recommend
from .core.workspace import Workspace

PREVIEW_ROWS = 10
_DATA_EXPORTERS = {
    ".csv": lambda df, path: df.to_csv(path, index=False),
    ".tsv": lambda df, path: df.to_csv(path, index=False, sep="\t"),
    ".json": lambda df, path: df.to_json(path, orient="records", indent=2),
    ".xlsx": lambda df, path: df.to_excel(path, index=False),
    ".parquet": lambda df, path: df.to_parquet(path),
}


# A token is a run of plain characters and/or quoted chunks: key='{"a": 1}' stays one token.
_TOKEN_PATTERN = re.compile(r"""(?:[^\s"']|"[^"]*"|'[^']*')+""")
_QUOTES = "'\""


def split_arguments(line: str) -> list[str]:
    """Split a command line into tokens, keeping Windows backslashes intact.

    ``shlex`` in POSIX mode treats backslashes as escapes (mangling Windows paths) and in
    non-POSIX mode cannot group a quote that starts mid-token. Surrounding quotes around a
    whole token or around a ``key=`` value are stripped: ``key='a b'`` and ``'a b'`` both work.
    """
    tokens = []
    for token in _TOKEN_PATTERN.findall(line):
        key, separator, value = ("", "", token) if token[0] in _QUOTES else token.partition("=")
        if len(value) > 1 and value[0] == value[-1] and value[0] in _QUOTES and value.count(value[0]) == 2:
            value = value[1:-1]
        tokens.append(f"{key}{separator}{value}")
    return tokens


def parse_key_values(arguments: list[str]) -> dict[str, Any]:
    """Turn ['a=1', 'cols=x,y', 'f=[1,2]'] into {'a': 1, 'cols': 'x,y', 'f': [1, 2]}."""
    parsed = {}
    for argument in arguments:
        key, separator, value = argument.partition("=")
        if not separator or not key:
            raise OperationError(f"Expected key=value, got '{argument}'.")
        parsed[key] = parse_cli_value(value)
    return parsed


def format_table(df: pd.DataFrame, rows: int = PREVIEW_ROWS) -> str:
    with pd.option_context("display.width", 160, "display.max_columns", 40, "display.max_colwidth", 40):
        return df.head(rows).to_string()


class CleanerShell(cmd.Cmd):
    intro = "PipeCleaner - interactive data cleaning and visualization. Type 'help' for commands, 'ops' to list cleaning operations."
    prompt = "pipecleaner> "

    def __init__(self, workspace: Workspace | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.workspace = workspace or Workspace()
        self.recommendations: list = []

    # ------------------------------------------------------------------ plumbing

    @property
    def dataset(self) -> Dataset:
        if self.workspace.active is None:
            raise OperationError("No dataset loaded. Use: load <path-or-url>")
        return self.workspace.active

    def onecmd(self, line: str) -> bool:
        """Run one command, reporting errors instead of crashing the REPL."""
        try:
            return super().onecmd(line)
        except OperationError as error:
            print(f"Error: {error}")
        except Exception as error:  # pandas/IO errors should not kill the session
            print(f"Error: {type(error).__name__}: {error}")
        return False

    def emptyline(self) -> bool:
        return False

    def default(self, line: str) -> None:
        """Any registered operation name is a command: ``<operation> key=value ...``."""
        name, *arguments = split_arguments(line)
        if name not in OPERATIONS:
            print(f"Unknown command '{name}'. Type 'help' or 'ops'.")
            return
        self._run_operation(name, parse_key_values(arguments))

    def _run_operation(self, name: str, params: dict[str, Any]) -> None:
        dataset = self.dataset
        rows_before, cells_before = len(dataset.current), len(dataset.changed_cells())
        removed_before = len(dataset.removed_rows())
        dataset.apply(name, **params)
        step_number = len(dataset.steps)
        print(f"Step {step_number}: {name}\n--- code ---\n{dataset.step_code(step_number)}\n------------")
        print(f"rows {rows_before} -> {len(dataset.current)}, columns: {len(dataset.current.columns)}")
        newly_removed = len(dataset.removed_rows()) - removed_before
        newly_changed = len(dataset.changed_cells()) - cells_before
        if newly_removed or newly_changed:
            print(f"quarantined: {newly_removed} row(s) removed, {newly_changed} cell(s) changed")

    # --------------------------------------------------------------------- loading

    def do_load(self, arg: str) -> None:
        """load <path-or-url>: load a CSV/TSV/JSON/Excel/Parquet file or URL as a new dataset."""
        if not arg.strip():
            raise OperationError(f"Usage: load <path-or-url>   (formats: {', '.join(SUPPORTED_EXTENSIONS)})")
        dataset = self.workspace.add_loaded(load_data(arg.strip()))
        print(f"Loaded '{dataset.name}': {dataset.current.shape[0]} rows x {dataset.current.shape[1]} columns")

    def do_datasets(self, arg: str) -> None:
        """datasets: list loaded datasets (* marks the active one)."""
        for name, dataset in self.workspace.datasets.items():
            marker = "*" if name == self.workspace.active_name else " "
            print(f"{marker} {name}: {dataset.current.shape[0]} rows, {len(dataset.steps)} step(s)")

    def do_use(self, arg: str) -> None:
        """use <name>: make a dataset active."""
        self.workspace.select(arg.strip())
        print(f"Active dataset: {arg.strip()}")

    # --------------------------------------------------------------------- viewing

    def do_show(self, arg: str) -> None:
        """show [rows]: preview the current table (default 10 rows)."""
        rows = int(arg) if arg.strip() else PREVIEW_ROWS
        print(format_table(self.dataset.current, rows))
        print(f"[{self.dataset.current.shape[0]} rows x {self.dataset.current.shape[1]} columns]")

    def do_original(self, arg: str) -> None:
        """original [rows]: preview the untouched original data."""
        print(format_table(self.dataset.original, int(arg) if arg.strip() else PREVIEW_ROWS))

    def do_info(self, arg: str) -> None:
        """info: column dtypes, null counts and unique counts."""
        current = self.dataset.current
        summary = pd.DataFrame(
            {"dtype": current.dtypes.astype(str), "nulls": current.isna().sum(), "unique": current.nunique()}
        )
        print(summary.to_string())

    def do_ops(self, arg: str) -> None:
        """ops: list every cleaning operation. Use 'help <operation>' for its parameters."""
        for name, operation in OPERATIONS.items():
            if not operation.analysis_only:
                print(f"  {name:<16} {operation.summary}")
        print("Analysis (never modifies the data): summary, counts, group_by, pivot, corr")

    def do_help(self, arg: str) -> None:
        """help [command|operation]: show help."""
        topic = arg.strip()
        if topic in OPERATIONS:
            operation = OPERATIONS[topic]
            print(f"{topic}: {operation.summary}\nUsage: {topic} key=value ...")
            for parameter, description in operation.parameter_docs.items():
                print(f"  {parameter:<14} {description}")
        else:
            super().do_help(arg)

    # --------------------------------------------------------------------- history

    def do_steps(self, arg: str) -> None:
        """steps: list the recorded cleaning steps."""
        for number, step in enumerate(self.dataset.steps, start=1):
            print(f"{number:>3}. {step.label()}")
        if not self.dataset.steps:
            print("No steps yet.")

    def do_undo(self, arg: str) -> None:
        """undo: undo the last step."""
        self.dataset.undo()
        print(f"Undone. {len(self.dataset.steps)} step(s) active.")

    def do_redo(self, arg: str) -> None:
        """redo: redo the last undone step."""
        self.dataset.redo()
        print(f"Redone. {len(self.dataset.steps)} step(s) active.")

    def do_reset(self, arg: str) -> None:
        """reset: go back to the original data (redo can bring steps back)."""
        self.dataset.reset()
        print("Reset to original data.")

    def do_remove(self, arg: str) -> None:
        """remove <step-number>: delete one step and replay the rest."""
        self.dataset.remove_step(int(arg) - 1)
        print(f"Removed step {arg}. {len(self.dataset.steps)} step(s) remain.")

    def do_save_steps(self, arg: str) -> None:
        """save_steps <file.json>: save the steps so they can be replayed later."""
        Path(arg.strip()).write_text(self.dataset.steps_to_json(), encoding="utf-8")
        print(f"Saved {len(self.dataset.steps)} step(s) to {arg.strip()}")

    def do_load_steps(self, arg: str) -> None:
        """load_steps <file.json>: replace the history with steps from a saved file."""
        self.dataset.load_steps_json(Path(arg.strip()).read_text(encoding="utf-8"))
        print(f"Loaded {len(self.dataset.steps)} step(s).")

    # -------------------------------------------------------------------- analysis
    # These only read the data. Add save_as=<name> to keep a result as a new dataset.

    def _show_analysis(self, result: analysis.AnalysisResult, options: dict[str, Any]) -> None:
        save_as = options.pop("save_as", None)
        if options:
            raise OperationError(f"Unknown option(s): {sorted(options)}")
        print(format_table(result.table, 60))
        print(f"[{len(result.table)} rows]\n--- code ---\n{result.code}\n------------")
        if save_as:
            dataset = self.workspace.save_analysis(self.dataset.name, str(save_as), result)
            print(f"Saved as new dataset '{dataset.name}' (now active). The source dataset is unchanged.")

    def do_summary(self, arg: str) -> None:
        """summary: column overview (types, nulls, distinct values) and numeric statistics
        (count, sum, mean, median, std, var, min, quartiles, max)."""
        current = self.dataset.current
        print(format_table(analysis.column_overview(current).table, 60))
        if analysis.numeric_columns(current):
            print()
            print(format_table(analysis.numeric_statistics(current).table, 60))

    def do_counts(self, arg: str) -> None:
        """counts <column> [limit=25] [nulls=true]: distinct values with counts and percent."""
        column, *rest = split_arguments(arg)
        options = parse_key_values(rest)
        result = analysis.value_counts(
            self.dataset.current, column, bool(options.get("nulls", False)), int(options.get("limit", 25)) or None
        )
        print(format_table(result.table, 60))
        print(f"[{self.dataset.current[column].nunique()} distinct values]")

    def do_group_by(self, arg: str) -> None:
        """group_by by=a,b aggregations='{"sales":["sum","mean"]}' [rows=true] [save_as=name]:
        summarize per group. Shows the result; does not change the dataset."""
        options = parse_key_values(split_arguments(arg))
        result = analysis.grouped_summary(
            self.dataset.current,
            as_list(options.pop("by", None)) or [],
            options.pop("aggregations", None) or {},
            bool(options.pop("rows", True)),
        )
        self._show_analysis(result, options)

    def do_pivot(self, arg: str) -> None:
        """pivot index=a columns=b values=c [aggfunc=sum] [save_as=name]: cross-tabulate.
        Shows the result; does not change the dataset."""
        options = parse_key_values(split_arguments(arg))
        result = analysis.pivot_summary(
            self.dataset.current,
            as_list(options.pop("index", None)) or [],
            as_list(options.pop("columns", None)) or [],
            as_list(options.pop("values", None)) or [],
            options.pop("aggfunc", "sum"),
        )
        self._show_analysis(result, options)

    def do_corr(self, arg: str) -> None:
        """corr [method=pearson|spearman|kendall]: correlation matrix of numeric columns."""
        options = parse_key_values(split_arguments(arg))
        print(format_table(analysis.correlations(self.dataset.current, options.get("method", "pearson")).table, 60))

    # ------------------------------------------------------------------ quarantine

    def do_quarantine(self, arg: str) -> None:
        """quarantine [rows|cells]: show rows removed / cells changed by cleaning steps."""
        dataset = self.dataset
        if arg.strip() in ("", "rows"):
            removed = dataset.removed_rows()
            print(f"Removed rows ({len(removed)}):\n{format_table(removed, 50) if len(removed) else '  none'}")
        if arg.strip() in ("", "cells"):
            changed = dataset.changed_cells()
            print(f"Changed cells ({len(changed)}):\n{format_table(changed, 50) if len(changed) else '  none'}")

    def do_restore(self, arg: str) -> None:
        """restore <n> [n ...]: move quarantined rows (by position in 'quarantine rows') back."""
        positions = [int(token) for token in arg.split()]
        self.dataset.restore_rows(positions)
        print(f"Restored {len(positions)} row(s).")

    # -------------------------------------------------------------- recommendations

    def do_recommend(self, arg: str) -> None:
        """recommend: suggest cleaning steps. Apply one with 'apply_rec <n>'."""
        self.recommendations = recommend(self.dataset.current)
        for number, item in enumerate(self.recommendations, start=1):
            print(f"{number:>3}. [{'!' * (4 - item.priority)}] {item.title}\n       {item.reason}")
        if not self.recommendations:
            print("No recommendations - the data looks clean.")

    def do_apply_rec(self, arg: str) -> None:
        """apply_rec <n|all>: apply recommendation n from the last 'recommend', or all of them."""
        if arg.strip().lower() == "all":
            outcome = apply_all_recommendations(self.dataset)
            for title in outcome.applied:
                print(f"  applied: {title}")
            for note in outcome.skipped:
                print(f"  skipped: {note}")
            print(f"Applied {len(outcome.applied)} suggestion(s); {len(self.dataset.steps)} step(s) total.")
            self.recommendations = []
            return
        number = int(arg)
        if not 1 <= number <= len(self.recommendations):
            raise OperationError("Run 'recommend' first, then pick one of the listed numbers.")
        item = self.recommendations[number - 1]
        self._run_operation(item.operation, item.params)
        self.recommendations = []  # numbering is stale once the data has changed

    # ------------------------------------------------------------------ combining

    def do_merge(self, arg: str) -> None:
        """merge <left> <right> <new_name> [how=inner] [left_on=a] [right_on=b] [on=key]."""
        left, right, new_name, *rest = split_arguments(arg)
        options = parse_key_values(rest)
        if "on" in options:
            options["left_on"] = options["right_on"] = options.pop("on")
        for key in ("left_on", "right_on"):
            if isinstance(options.get(key), str):
                options[key] = [part.strip() for part in options[key].split(",")]
        dataset = self.workspace.merge(left, right, new_name, **options)
        print(f"Created '{dataset.name}': {dataset.current.shape[0]} rows x {dataset.current.shape[1]} columns")

    def do_concat(self, arg: str) -> None:
        """concat <name> <name> ... <new_name> [axis=0] [join=outer]: stack datasets."""
        names, options = [], {}
        for token in split_arguments(arg):
            (options.update(parse_key_values([token])) if "=" in token else names.append(token))
        *sources, new_name = names
        dataset = self.workspace.concat(sources, new_name, **options)
        print(f"Created '{dataset.name}': {dataset.current.shape[0]} rows x {dataset.current.shape[1]} columns")

    # ----------------------------------------------------------- code / export / plot

    def do_code(self, arg: str) -> None:
        """code [step]: show the pandas script for everything done (or for one step)."""
        if arg.strip():
            print(self.dataset.step_code(int(arg)))
        else:
            print(self.dataset.export_script())

    def do_export(self, arg: str) -> None:
        """export <file>: save the current data (.csv .tsv .json .xlsx .parquet) or script (.py)."""
        path = Path(arg.strip().strip('"'))
        if path.suffix == ".py":
            path.write_text(self.dataset.export_script(), encoding="utf-8")
        elif path.suffix in _DATA_EXPORTERS:
            _DATA_EXPORTERS[path.suffix](self.dataset.current, path)
        else:
            raise OperationError(f"Use one of: .py {' '.join(_DATA_EXPORTERS)}")
        print(f"Wrote {path}")

    def do_plot(self, arg: str) -> None:
        """plot <kind> key=value ... [out=file.html|png] [title=..]: create a Plotly Express chart.

        Kinds: see 'plot' with no arguments. Example: plot scatter x=age y=income color=city out=a.html
        color/size/symbol take a column or a fixed value: plot scatter x=a y=b color=red size=12
        """
        tokens = split_arguments(arg)
        if not tokens:
            for kind, specs in PLOT_KINDS.items():
                print(f"  {kind:<20} " + ", ".join(spec.name + ("*" if spec.required else "") for spec in specs))
            print("(* = required)")
            return
        kind, options = tokens[0], parse_key_values(tokens[1:])
        output = options.pop("out", None)
        title = options.pop("title", None)
        if kind not in PLOT_KINDS:
            raise OperationError(f"Unknown plot kind '{kind}'. Run 'plot' to list them.")
        options = resolve_fixed_arguments(self.dataset.current, kind, options)  # color=red -> fixed color
        figure = build_figure(self.dataset.current, kind, options, title=title)
        print("--- code ---\n" + plot_code(kind, options, title) + "\n------------")
        if output is None:
            figure.show()
        elif str(output).endswith(".html"):
            figure.write_html(output)
            print(f"Wrote {output}")
        else:
            figure.write_image(output)  # needs the optional 'kaleido' package
            print(f"Wrote {output}")

    # ------------------------------------------------------------------------ exit

    def do_quit(self, arg: str) -> bool:
        """quit: leave the program."""
        return True

    do_exit = do_quit

    def do_EOF(self, arg: str) -> bool:
        print()
        return True


def launch_streamlit_app() -> int:
    """Start the Streamlit version of the app."""
    app_path = Path(__file__).with_name("streamlit_app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipecleaner", description="Interactive data cleaning and visualization."
    )
    parser.add_argument("sources", nargs="*", help="files or URLs to load at startup")
    parser.add_argument("-c", "--command", action="append", default=[], help="run a command (repeatable)")
    parser.add_argument("-f", "--file", type=Path, help="run commands from a text file, one per line")
    parser.add_argument("-i", "--interactive", action="store_true", help="stay interactive after -c/-f")
    parser.add_argument("--app", action="store_true", help="launch the Streamlit app instead")
    args = parser.parse_args(argv)

    if args.app:
        return launch_streamlit_app()

    shell = CleanerShell()
    scripted = list(args.command)
    if args.file:
        scripted += [line.strip() for line in args.file.read_text(encoding="utf-8").splitlines()]
    for source in args.sources:
        shell.onecmd(f"load {source}")
    for line in filter(None, scripted):
        if line.startswith("#"):
            continue
        print(f"{shell.prompt}{line}")
        shell.onecmd(line)
    if not scripted or args.interactive:
        shell.cmdloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
