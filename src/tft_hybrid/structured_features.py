import os
import numpy as np
import pandas as pd

from .config import Paths, MEGA_CAP_5

ANCHOR_SHIFT_DAYS = 5


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = numer / denom.replace({0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)


def safe_log1p(x):
    return np.log1p(np.clip(x, a_min=0, a_max=None))


def engineer_one_file(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    anchor = df["date"] - pd.Timedelta(days=ANCHOR_SHIFT_DAYS)
    df["year"] = anchor.dt.year
    df["quarter"] = anchor.dt.quarter.apply(lambda x: f"Q{x}")
    df["quarter_int"] = anchor.dt.quarter
    df["Year_Quarter"] = df["year"].astype(str) + df["quarter"]

    dollar_cols = [
        "revenue", "costOfRevenue", "grossProfit", "operatingExpenses", "ebitda",
        "operatingIncome", "incomeBeforeTax", "netIncome", "totalAssets", "totalEquity",
    ]
    if "researchAndDevelopmentExpenses" in df.columns:
        df["rnd"] = df["researchAndDevelopmentExpenses"]
        dollar_cols.append("rnd")
    if "sellingGeneralAndAdministrativeExpenses" in df.columns:
        df["snaExpenses"] = df["sellingGeneralAndAdministrativeExpenses"]
        dollar_cols.append("snaExpenses")

    for col in dollar_cols:
        if col in df.columns:
            df[col] = df[col] / 1_000_000

    df = df.sort_values(["year", "quarter_int"]).reset_index(drop=True)
    df["revenue_log"] = safe_log1p(df["revenue"])
    df["revenue_log_lag1"] = df["revenue_log"].shift(1)
    df["revenue_log_lag4"] = df["revenue_log"].shift(4)

    if {"operatingIncome", "revenue"}.issubset(df.columns):
        df["operatingIncomeRatio"] = safe_div(df["operatingIncome"], df["revenue"])
    if {"grossProfit", "revenue"}.issubset(df.columns):
        df["grossProfitRatio"] = safe_div(df["grossProfit"], df["revenue"])
    if {"netIncome", "revenue"}.issubset(df.columns):
        df["netIncomeRatio"] = safe_div(df["netIncome"], df["revenue"])

    for col in ["revenue_log", "operatingIncome", "netIncome"]:
        if col in df.columns:
            df[f"{col}_yoy"] = safe_div(df[col], df[col].shift(4)) - 1.0

    if "totalAssets" in df.columns:
        df["totalAssets_lag1"] = df["totalAssets"].shift(1)
        df["totalAssets_lag1_log"] = safe_log1p(df["totalAssets_lag1"])
    if "totalEquity" in df.columns:
        df["totalEquity_lag1"] = df["totalEquity"].shift(1)
        df["totalEquity_lag1_log"] = safe_log1p(df["totalEquity_lag1"])

    df["gics_sectors"] = df.get("sector", pd.NA)
    return df


def main():
    paths = Paths()
    os.makedirs(paths.financial_data_dir, exist_ok=True)
    processed_count = 0
    for ticker in MEGA_CAP_5:
        raw_fp = os.path.join(paths.financial_data_dir, f"{ticker}.csv")
        if not os.path.exists(raw_fp):
            print(f"[WARN] Raw file for {ticker} not found at {raw_fp}. Skipping.")
            continue
        df = pd.read_csv(raw_fp)
        feat = engineer_one_file(df)
        out_fp = os.path.join(paths.financial_data_dir, f"{ticker}_feature.csv")
        feat.to_csv(out_fp, index=False)
        print(f"Processed {ticker}: {len(feat)} quarters -> {out_fp}")
        processed_count += 1
    if processed_count == 0:
        print("[ERROR] No files were processed. Run fetch_megacap5.py first.")
