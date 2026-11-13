"""Load raw AI4I machine readings."""

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path("data/ai4i2020.csv")


def load_readings(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the locally downloaded AI4I CSV."""
    return pd.read_csv(path)
