"""Session state and small helpers shared by the Streamlit views."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from data_cleaner.core.workspace import Workspace

_WORKSPACE_KEY = "workspace"
_FLASH_KEY = "flash_message"


def get_workspace() -> Workspace:
    """The current user's workspace (created on first use, kept across reruns)."""
    if _WORKSPACE_KEY not in st.session_state:
        st.session_state[_WORKSPACE_KEY] = Workspace()
    return st.session_state[_WORKSPACE_KEY]


def flash(message: str) -> None:
    """Remember a message to show after the next rerun (a rerun would erase it otherwise)."""
    st.session_state[_FLASH_KEY] = message


def show_flash() -> None:
    message = st.session_state.pop(_FLASH_KEY, None)
    if message:
        st.success(message, icon=":material/check_circle:")


def run_action(action: Callable[[], Any], success_message: str) -> bool:
    """Run ``action``; on success flash a message and rerun, on failure show the error."""
    try:
        action()
    except Exception as error:  # surface pandas/IO/validation errors in the UI instead of a traceback
        st.error(str(error) if str(error) else type(error).__name__, icon=":material/error:")
        return False
    flash(success_message)
    st.rerun()
    return True  # unreachable: st.rerun() stops the script, kept for type checkers
