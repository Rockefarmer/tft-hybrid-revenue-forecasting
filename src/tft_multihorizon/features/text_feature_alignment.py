import pandas as pd


def merge_text_features(panel_df: pd.DataFrame, text_df: pd.DataFrame, on=None) -> pd.DataFrame:
    on = on or ["ticker", "quarter"]
    return panel_df.merge(text_df, on=on, how="left")
