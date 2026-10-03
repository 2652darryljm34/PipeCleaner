"""Sidebar: loading data, choosing the active dataset, undo/redo and downloads."""

from __future__ import annotations

import io

import streamlit as st

from data_cleaner.core.loaders import SUPPORTED_EXTENSIONS, load_data
from data_cleaner.core.workspace import Workspace

from .state import run_action


def render_sidebar(workspace: Workspace) -> None:
    with st.sidebar:
        st.header("Data")
        _render_loaders(workspace)
        if workspace.active is not None:
            st.divider()
            _render_dataset_controls(workspace)


def _render_loaders(workspace: Workspace) -> None:
    upload_tab, url_tab = st.tabs(["Upload", "URL"])
    with upload_tab:
        files = st.file_uploader(
            "Upload files", type=list(SUPPORTED_EXTENSIONS), accept_multiple_files=True, label_visibility="collapsed"
        )
        loaded_ids = st.session_state.setdefault("loaded_upload_ids", set())
        for uploaded in files or []:
            if uploaded.file_id in loaded_ids:
                continue  # already in the workspace; uploaders keep their files across reruns
            try:
                workspace.add_loaded(load_data(uploaded, filename=uploaded.name))
            except Exception as error:
                st.error(f"Could not read {uploaded.name}: {error}")
            loaded_ids.add(uploaded.file_id)
    with url_tab:
        url = st.text_input("Link to a CSV, JSON, Excel or Parquet file", placeholder="https://.../data.csv")
        if st.button("Load from URL", disabled=not url.strip()):
            with st.spinner("Downloading..."):
                run_action(lambda: workspace.add_loaded(load_data(url.strip())), "Loaded data from URL.")


def _render_dataset_controls(workspace: Workspace) -> None:
    names = list(workspace.datasets)
    chosen = st.selectbox("Active dataset", names, index=names.index(workspace.active_name))
    if chosen != workspace.active_name:
        workspace.select(chosen)
        st.rerun()
    dataset = workspace.active

    undo_column, redo_column, reset_column = st.columns(3)
    if undo_column.button("Undo", disabled=not dataset.can_undo, icon=":material/undo:", width="stretch"):
        run_action(dataset.undo, "Undone.")
    if redo_column.button("Redo", disabled=not dataset.can_redo, icon=":material/redo:", width="stretch"):
        run_action(dataset.redo, "Redone.")
    if reset_column.button("Reset", disabled=not dataset.can_undo, icon=":material/restart_alt:", width="stretch", help="Back to the original data (Redo can bring the steps back)."):
        run_action(dataset.reset, "Reset to the original data.")
    st.caption(f"{len(dataset.steps)} step(s) applied")

    st.download_button("Download CSV", lambda: dataset.current.to_csv(index=False), file_name=f"{dataset.name}.csv", mime="text/csv", icon=":material/download:", width="stretch")
    st.download_button("Download Excel", lambda: _to_excel_bytes(dataset), file_name=f"{dataset.name}.xlsx", icon=":material/download:", width="stretch")
    if st.button("Remove dataset", icon=":material/delete:", width="stretch"):
        run_action(lambda: workspace.remove(dataset.name), f"Removed '{dataset.name}'.")


def _to_excel_bytes(dataset) -> bytes:
    buffer = io.BytesIO()
    dataset.current.to_excel(buffer, index=False)
    return buffer.getvalue()
