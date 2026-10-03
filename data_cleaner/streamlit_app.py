"""Streamlit entry point:  streamlit run data_cleaner/streamlit_app.py

Kept as a plain script on purpose; all logic lives in ``data_cleaner.core`` and ``data_cleaner.ui``.
"""

import sys
from pathlib import Path

# Make `import data_cleaner` work when launched straight from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from data_cleaner.ui.analysis_view import render_analyze  # noqa: E402
from data_cleaner.ui.sidebar import render_sidebar  # noqa: E402
from data_cleaner.ui.state import get_workspace, show_flash  # noqa: E402
from data_cleaner.ui.views import (  # noqa: E402
    render_clean,
    render_code,
    render_combine,
    render_data,
    render_plot,
    render_quarantine,
    render_recommendations,
)

st.set_page_config(page_title="PipeCleaner", page_icon=":material/cleaning_services:", layout="wide")

workspace = get_workspace()
render_sidebar(workspace)

st.title("PipeCleaner")
st.caption("Interactive data cleaning and visualization")
show_flash()

dataset = workspace.active
if dataset is None:
    st.info("Upload a file or paste a URL in the sidebar to get started.", icon=":material/upload_file:")
    st.stop()

SECTIONS = {
    ":material/table: Data": lambda: render_data(dataset),
    ":material/lightbulb: Recommendations": lambda: render_recommendations(dataset),
    ":material/cleaning_services: Clean": lambda: render_clean(dataset),
    ":material/analytics: Analyze": lambda: render_analyze(workspace, dataset),
    ":material/merge: Combine": lambda: render_combine(workspace),
    ":material/inventory_2: Quarantine": lambda: render_quarantine(dataset),
    ":material/bar_chart: Plot": lambda: render_plot(dataset),
    ":material/code: Code": lambda: render_code(dataset),
}
section = st.segmented_control(
    "Section", list(SECTIONS), default=next(iter(SECTIONS)), key="section", label_visibility="collapsed"
)
SECTIONS[section or next(iter(SECTIONS))]()  # only the selected section does any work
