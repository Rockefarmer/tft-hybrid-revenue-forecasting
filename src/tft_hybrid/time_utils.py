import re
import numpy as np
import pandas as pd

from .config import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST


def parse_yq_str(yq: str):
    s = str(yq).strip().upper()
    m = re.search(r"(\d{4})\s*[-/ ]?\s*Q\s*([1-4])", s)
    if not m:
        raise ValueError(f"Cannot parse quarter string: {yq!r}")
    return int(m.group(1)), int(m.group(2))


def yq_scalar(year: int, quarter: int) -> int:
    return int(year) * 4 + (int(quarter) - 1)


def yq_key_vec(year_series, quarter_series) -> pd.Series:
    y = pd.to_numeric(year_series, errors="coerce")
    q = pd.to_numeric(quarter_series, errors="coerce")
    return (y * 4 + (q - 1)).astype("Int64")


def ensure_year_quarter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "year" not in df.columns:
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            df["year"] = dt.dt.year
        elif "Year_Quarter" in df.columns:
            y = df["Year_Quarter"].astype(str).str.extract(r"(\d{4})", expand=False)
            df["year"] = pd.to_numeric(y, errors="coerce")
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "quarter_int" not in df.columns:
        if "quarter" in df.columns:
            q = pd.to_numeric(df["quarter"], errors="coerce")
            if q.dropna().between(0, 3).all():
                q = q.replace({0: 1, 1: 2, 2: 3, 3: 4})
            df["quarter_int"] = q
        elif "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            df["quarter_int"] = dt.dt.quarter
    else:
        df["quarter_int"] = pd.to_numeric(df["quarter_int"], errors="coerce")
        if df["quarter_int"].dropna().between(0, 3).all():
            df["quarter_int"] = df["quarter_int"].replace({0: 1, 1: 2, 2: 3, 3: 4})
    return df


def attach_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "year", "quarter_int"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("ticker").cumcount()
    return df


def apply_fixed_splits(df: pd.DataFrame) -> pd.DataFrame:
    bounds = []
    for start, end in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        y1, q1 = parse_yq_str(start)
        y2, q2 = parse_yq_str(end)
        bounds.append((yq_scalar(y1, q1), yq_scalar(y2, q2)))
    (train_lo, train_hi), (val_lo, val_hi), (test_lo, test_hi) = bounds

    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["quarter_int"] = pd.to_numeric(out["quarter_int"], errors="coerce")
    out = out[out["year"].notna() & out["quarter_int"].notna()]
    out["yq_key"] = yq_key_vec(out["year"], out["quarter_int"])
    out["split"] = np.where((out["yq_key"] >= train_lo) & (out["yq_key"] <= train_hi), "train",
                      np.where((out["yq_key"] >= val_lo) & (out["yq_key"] <= val_hi), "val",
                      np.where((out["yq_key"] >= test_lo) & (out["yq_key"] <= test_hi), "test", "drop")))
    return out[out["split"] != "drop"].copy()
