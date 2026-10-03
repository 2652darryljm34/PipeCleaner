import io

import pandas as pd
import pytest

from data_cleaner.cli import CleanerShell, parse_key_values


@pytest.fixture
def shell(tmp_path, messy_frame):
    path = tmp_path / "messy.csv"
    messy_frame.to_csv(path, index=False)
    output = io.StringIO()
    shell = CleanerShell(stdout=output)
    shell.onecmd(f"load {path}")
    return shell


def test_parse_key_values_handles_json_lists_and_text():
    assert parse_key_values(["a=1", "cols=x,y", 'f=["a","b"]', "flag=true"]) == {
        "a": 1,
        "cols": "x,y",
        "f": ["a", "b"],
        "flag": True,
    }


def test_operation_commands_apply_and_print_code(shell, capsys):
    shell.onecmd("drop_nulls subset=age,Name")
    shell.onecmd("rename_columns mapping='{\"city\": \"town\"}'")
    output = capsys.readouterr().out
    assert "df = df.dropna(subset=['age', 'Name'], how='any')" in output
    assert "quarantined: 3 row(s) removed" in output
    assert "town" in shell.dataset.current.columns


def test_errors_are_reported_not_raised(shell, capsys):
    shell.onecmd("drop_columns columns=nope")
    shell.onecmd("not_a_command")
    shell.onecmd("drop_nulls oops")
    output = capsys.readouterr().out
    assert output.count("Error:") == 2 and "Unknown command" in output


def test_recommend_apply_undo_export(shell, tmp_path, capsys):
    shell.onecmd("recommend")
    assert shell.recommendations
    shell.onecmd("apply_rec 1")
    shell.onecmd("undo")
    assert shell.dataset.steps == []
    shell.onecmd("drop_duplicates")
    shell.onecmd(f"export {tmp_path / 'clean.csv'}")
    shell.onecmd(f"export {tmp_path / 'clean.py'}")
    assert len(pd.read_csv(tmp_path / "clean.csv")) == len(shell.dataset.current)
    assert "def build_messy" in (tmp_path / "clean.py").read_text()


def test_plot_writes_html_with_code(shell, tmp_path, capsys):
    out = tmp_path / "plot.html"
    shell.onecmd(f"plot bar x=city y=age out={out}")
    assert out.exists() and "px.bar" in capsys.readouterr().out


def test_plot_accepts_fixed_color_and_size(shell, tmp_path, capsys):
    out = tmp_path / "fixed.html"
    shell.onecmd(f"plot scatter x=age y=age color=red size=14 out={out}")
    output = capsys.readouterr().out
    assert out.exists() and "marker_color='red'" in output and "marker_size=14" in output
