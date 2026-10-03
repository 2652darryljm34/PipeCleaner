"""Load tabular data from an uploaded file, a local path or a URL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable
from urllib.parse import urlparse

import pandas as pd

# extension -> (pandas reader, reader keyword arguments)
_READERS: dict[str, tuple[Callable[..., pd.DataFrame], dict[str, Any]]] = {
    "csv": (pd.read_csv, {}),
    "txt": (pd.read_csv, {}),
    "tsv": (pd.read_csv, {"sep": "\t"}),
    "json": (pd.read_json, {}),
    "xlsx": (pd.read_excel, {}),
    "xls": (pd.read_excel, {}),
    "parquet": (pd.read_parquet, {}),
}
SUPPORTED_EXTENSIONS = tuple(_READERS)
_DEFAULT_EXTENSION = "csv"
# Some servers reject pandas' default urllib user agent.
_URL_HEADERS = {"User-Agent": "Mozilla/5.0 (PipeCleaner)"}


@dataclass
class LoadedData:
    """A freshly loaded table plus the pandas code that reproduces the load."""

    dataframe: pd.DataFrame
    origin_code: str
    suggested_name: str


def is_url(text: str) -> bool:
    return urlparse(text.strip()).scheme in {"http", "https"}


def load_data(source: str | Path | BinaryIO, filename: str | None = None) -> LoadedData:
    """Load ``source`` (URL, path or open file object) into a DataFrame.

    ``filename`` names an uploaded file object so its format can be detected.
    """
    if isinstance(source, (str, Path)):
        location = str(source).strip().strip('"')
        label = urlparse(location).path if is_url(location) else location
    else:
        location, label = source, filename or getattr(source, "name", "")

    extension = PurePosixPath(str(label).replace("\\", "/")).suffix.lstrip(".").lower()
    if extension not in _READERS:
        extension = _DEFAULT_EXTENSION  # unknown/missing extension: assume delimited text
    reader, reader_kwargs = _READERS[extension]
    if isinstance(location, str) and is_url(location):
        reader_kwargs = {**reader_kwargs, "storage_options": _URL_HEADERS}

    dataframe = reader(location, **reader_kwargs)

    display_name = location if isinstance(location, str) else (filename or "uploaded_file")
    code_kwargs = {k: v for k, v in reader_kwargs.items() if k != "storage_options"}
    arguments = "".join(f", {key}={value!r}" for key, value in code_kwargs.items())
    origin_code = f"df = pd.{reader.__name__}({display_name!r}{arguments})"
    if not isinstance(location, (str, Path)):
        origin_code = f"# Uploaded file: place it next to this script.\n{origin_code}"

    stem = PurePosixPath(str(display_name).replace("\\", "/").split("?")[0]).stem
    return LoadedData(dataframe, origin_code, stem or "dataset")
