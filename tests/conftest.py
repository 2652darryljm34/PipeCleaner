import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def messy_frame() -> pd.DataFrame:
    """A small frame with duplicates, nulls, text noise and numbers stored as text."""
    return pd.DataFrame(
        {
            "Name": ["Ann ", "bob", "Bob", "Cy", "Cy", None],
            "age": [25, 31, 31, np.nan, np.nan, 40],
            "score": ["1", "2", "x", "4", "4", "6"],
            "city": ["NY", "LA", "LA", "SF", "SF", "NY"],
        }
    )
