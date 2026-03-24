import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "revenue" in out.columns:
        out["revenue_log1p"] = out["revenue"].clip(lower=0).map(lambda x: __import__("math").log1p(x))
    return out
