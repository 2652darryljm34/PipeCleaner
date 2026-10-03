# PipeCleaner

Interactive data cleaning and visualization, built on pandas, as a **Streamlit app** and a **CLI REPL**.
Both are thin front ends over one framework-free core (`data_cleaner/core`), so they behave identically.

## Run it

```bash
pip install -r requirements.txt
streamlit run data_cleaner/streamlit_app.py      # the app
python -m data_cleaner.cli [file-or-url ...]     # the REPL  (or: pip install -e . ; pipecleaner)
python -m data_cleaner.cli data.csv -c "drop_duplicates" -c "export clean.csv"   # one-shot
```

## Features

| Need | Where |
|---|---|
| Upload file / load URL (csv, tsv, json, xlsx, parquet) | sidebar / `load` |
| Original copy preserved | Data > Original, `original` |
| Quarantine: removed rows + changed cells, with step and reason; restore rows | Quarantine, `quarantine`, `restore` |
| Recommended cleaning steps, one click to apply | Recommendations, `recommend` / `apply_rec` |
| Filter, rename, drop duplicates/nulls, convert dtypes, bins, fill/replace, group by, pivot, stack, melt, add/remove columns | Clean, or the operation name in the REPL (`ops`) |
| Edit cells, delete/add rows in the grid | Data (editable table) |
| Merge / concat datasets | Combine, `merge` / `concat` |
| Any Plotly Express plot type | Plot, `plot` |
| See the code that ran; export a standalone script; undo/redo; edit or remove any step | Code, `code`, `export x.py`, `undo` |

## How it works

* A **Dataset** is the untouched original plus a list of **steps** (`operation` + JSON params). The current table is the original replayed through the steps, which is what makes undo/redo, editing a step and removing a step safe.
* An **Operation** (`core/operations/`) has `apply` (pandas) and `to_code` (the same thing as readable pandas). Tests check both give identical results.
* Merged/concatenated datasets are snapshots of their sources at the time of combining.

## Tests

`python -m pytest` (operations, history, quarantine, scripts, CLI, and headless Streamlit `AppTest`s).
