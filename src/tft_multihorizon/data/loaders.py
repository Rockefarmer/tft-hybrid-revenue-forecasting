from pathlib import Path
import pandas as pd


def load_panel(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
