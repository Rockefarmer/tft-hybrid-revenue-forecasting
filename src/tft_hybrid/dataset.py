import os
import glob
import numpy as np
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet

from .config import Paths, MEGA_CAP_5
from .time_utils import ensure_year_quarter, attach_time_idx, apply_fixed_splits
from .merge_text_features import merge_finbert_features, merge_llama3_features, canonical_ticker


def safe_log1p(x):
    return np.log1p(np.clip(x, a_min=0, a_max=None))


def load_structured_panel(data_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*_feature.csv"))
    if not files:
        raise RuntimeError(f"No *_feature.csv files found in {data_dir}")
    frames = []
    for fp in files:
        fname = os.path.basename(fp).upper()
        if not any(f"{ticker}" in fname for ticker in MEGA_CAP_5):
            continue
        frames.append(pd.read_csv(fp))
    if not frames:
        raise RuntimeError("No readable feature files found for Mega-Cap 5")
    data = pd.concat(frames, ignore_index=True)
    if "ticker" not in data.columns:
        if "TICKER" in data.columns:
            data = data.rename(columns={"TICKER": "ticker"})
        else:
            raise ValueError("No ticker column found")
    data["ticker"] = data["ticker"].astype(str).map(canonical_ticker)
    data = data[data["ticker"].isin([canonical_ticker(t) for t in MEGA_CAP_5])].copy()
    data = ensure_year_quarter(data)
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["quarter_int"] = pd.to_numeric(data["quarter_int"], errors="coerce")
    data["Year_Quarter"] = data["year"].astype("Int64").astype(str) + "Q" + data["quarter_int"].astype("Int64").astype(str)
    if "revenue" not in data.columns:
        raise ValueError("Column 'revenue' is required.")
    if "revenue_log" not in data.columns:
        data["revenue_log"] = safe_log1p(data["revenue"])
    if "revenue_log_lag1" not in data.columns:
        data = data.sort_values(["ticker", "year", "quarter_int"])
        data["revenue_log_lag1"] = data.groupby("ticker")["revenue_log"].shift(1)
    if "revenue_log_lag4" not in data.columns:
        data = data.sort_values(["ticker", "year", "quarter_int"])
        data["revenue_log_lag4"] = data.groupby("ticker")["revenue_log"].shift(4)
    return data


def choose_known_future_cols(df: pd.DataFrame, text_source: str):
    known = ["year", "quarter_int"]
    for a, b in [("totalAssets_lag1_log", "totalAssets_lag1"), ("totalEquity_lag1_log", "totalEquity_lag1")]:
        if a in df.columns:
            known.append(a)
        elif b in df.columns:
            known.append(b)
    if text_source == "finbert":
        known.extend([c for c in df.columns if c.startswith("sent_") and c.endswith("_ffill")])
    elif text_source == "llama3":
        known.extend([c for c in ["sent_net_mean_ffill", "sent_fli_count_ffill"] if c in df.columns])
    return known


def build_feature_lists(df: pd.DataFrame, text_source: str):
    static_categoricals = [c for c in ["ticker", "gics_sectors"] if c in df.columns]
    known_reals = choose_known_future_cols(df, text_source)
    base_unknown = [
        "revenue_log", "revenue_log_lag1", "revenue_log_lag4",
        "grossProfit", "costOfRevenue", "operatingExpenses", "snaExpenses",
        "ebitda", "operatingIncome", "incomeBeforeTax", "netIncome",
        "totalAssets", "totalEquity", "grossProfitRatio", "operatingIncomeRatio", "netIncomeRatio",
    ]
    unknown = [c for c in base_unknown if c in df.columns]
    unknown += [c for c in df.columns if c.endswith("_yoy")]
    if text_source == "finbert":
        unknown += [c for c in df.columns if c.startswith("sent_") and not c.endswith("_ffill")]
    elif text_source == "llama3":
        unknown += [c for c in ["sent_net_mean", "sent_fli_count"] if c in df.columns]
    unknown = [c for c in dict.fromkeys(unknown) if c not in known_reals]
    return static_categoricals, known_reals, [], unknown


def sanitize_encoder_features(df: pd.DataFrame, encoder_cols: list):
    out = df.copy()
    added_flags = []
    out[encoder_cols] = out[encoder_cols].replace([np.inf, -np.inf], np.nan)
    for col in encoder_cols:
        na = out[col].isna()
        if not na.any():
            continue
        flag = f"{col}_nanflag"
        out[flag] = na.astype(np.int8)
        added_flags.append(flag)
        if col.endswith("_yoy"):
            out.loc[na, col] = 0.0
        else:
            med = out.groupby("ticker")[col].transform("median")
            out.loc[na, col] = med[na]
            na2 = out[col].isna()
            if na2.any():
                out.loc[na2, col] = 0.0
    return out, added_flags


def prepare_hybrid_frames(text_source: str):
    paths = Paths()
    df = load_structured_panel(paths.financial_data_dir)
    if text_source == "finbert":
        df = merge_finbert_features(df, paths.finbert_quarterly_csv)
    elif text_source == "llama3":
        df = merge_llama3_features(df, paths.llama3_quarterly_csv)
    else:
        raise ValueError(f"Unsupported text_source: {text_source}")
    df = attach_time_idx(df)
    df = apply_fixed_splits(df)

    static_categoricals, known_reals, known_categoricals, unknown_reals = build_feature_lists(df, text_source)
    req_lags = [c for c in ["revenue_log", "revenue_log_lag1", "revenue_log_lag4"] if c in df.columns]

    def prune(split_df):
        return split_df.dropna(subset=req_lags).copy()

    train_base = prune(df.loc[df["split"].eq("train")])
    val_base = prune(df.loc[df["split"].eq("val")])
    test_base = prune(df.loc[df["split"].eq("test")])

    max_encoder = 12 if text_source == "finbert" else 4
    val_history = train_base.groupby("ticker").tail(max_encoder)
    val_ext = pd.concat([val_history, val_base]).sort_values(["ticker", "time_idx"])
    tv_combined = pd.concat([train_base, val_base]).sort_values(["ticker", "time_idx"])
    test_history = tv_combined.groupby("ticker").tail(max_encoder)
    test_ext = pd.concat([test_history, test_base]).sort_values(["ticker", "time_idx"])

    encoder_cols = list(dict.fromkeys(["revenue_log"] + unknown_reals))
    train_df, flags_train = sanitize_encoder_features(train_base, encoder_cols)
    val_df, flags_val = sanitize_encoder_features(val_ext, encoder_cols)
    test_df, flags_test = sanitize_encoder_features(test_ext, encoder_cols)
    flag_cols = sorted(set(flags_train) | set(flags_val) | set(flags_test))
    for col in flag_cols:
        if col not in train_df.columns: train_df[col] = 0
        if col not in val_df.columns: val_df[col] = 0
        if col not in test_df.columns: test_df[col] = 0
    for col in encoder_cols:
        if col not in train_df.columns: train_df[col] = 0.0
        if col not in val_df.columns: val_df[col] = 0.0
        if col not in test_df.columns: test_df[col] = 0.0
    unknown_extended = list(dict.fromkeys(unknown_reals + flag_cols))
    return {
        "full": df,
        "train_base": train_base,
        "val_base": val_base,
        "test_base": test_base,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "static_categoricals": static_categoricals,
        "known_reals": known_reals,
        "known_categoricals": known_categoricals,
        "unknown_reals_extended": unknown_extended,
        "max_encoder": max_encoder,
    }


def build_tsdatasets(bundle: dict, max_prediction_length: int = 4):
    training = TimeSeriesDataSet(
        bundle["train_df"],
        time_idx="time_idx",
        target="revenue_log",
        group_ids=["ticker"],
        max_encoder_length=bundle["max_encoder"],
        max_prediction_length=max_prediction_length,
        static_categoricals=bundle["static_categoricals"],
        static_reals=[],
        time_varying_known_categoricals=bundle["known_categoricals"],
        time_varying_known_reals=bundle["known_reals"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=list(dict.fromkeys(["revenue_log"] + bundle["unknown_reals_extended"])),
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = training.from_dataset(training, bundle["val_df"])
    testing = training.from_dataset(training, bundle["test_df"])
    return training, validation, testing
