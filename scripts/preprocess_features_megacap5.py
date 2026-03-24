import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# === Paths ===
# NEW CONSOLIDATED DIRECTORY (Reads and writes to the same folder)
DATA_DIR = r"C:\ThesisResearch\finbert_tft\data\financial_data"
os.makedirs(DATA_DIR, exist_ok=True)

MEGA_CAP_5 = ["AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "NVDA"]

ANCHOR_SHIFT_DAYS = 5
_QMAP = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = numer / denom.replace({0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)

def _safe_log1p(x):
    return np.log1p(np.clip(x, a_min=0, a_max=None))

def main():
    processed_count = 0
    
    for ticker in MEGA_CAP_5:
        # Explicitly look for the raw file (e.g., AAPL.csv) to avoid reading AAPL_feature.csv
        raw_fp = os.path.join(DATA_DIR, f"{ticker}.csv")
        
        if not os.path.exists(raw_fp):
            print(f"[WARN] Raw file for {ticker} not found at {raw_fp}. Skipping.")
            continue
            
        df = pd.read_csv(raw_fp)
        
        # --- Preprocessing Step ---
        df['date'] = pd.to_datetime(df['date'])
        anchor = df['date'] - pd.Timedelta(days=ANCHOR_SHIFT_DAYS)
        df['year'] = anchor.dt.year
        df['quarter'] = anchor.dt.quarter.apply(lambda x: f"Q{x}")
        df['quarter_int'] = anchor.dt.quarter
        df['Year_Quarter'] = df['year'].astype(str) + df['quarter']

        # Downscale to millions
        dollar_cols = [
            'revenue', 'costOfRevenue', 'grossProfit', 'operatingExpenses', 
            'ebitda', 'operatingIncome', 'incomeBeforeTax', 'netIncome', 
            'totalAssets', 'totalEquity'
        ]
        # Handle alternate FMP names for R&D and SG&A
        if 'researchAndDevelopmentExpenses' in df.columns:
            df['rnd'] = df['researchAndDevelopmentExpenses']
            dollar_cols.append('rnd')
        if 'sellingGeneralAndAdministrativeExpenses' in df.columns:
            df['snaExpenses'] = df['sellingGeneralAndAdministrativeExpenses']
            dollar_cols.append('snaExpenses')

        for col in dollar_cols:
            if col in df.columns:
                df[col] = df[col] / 1_000_000

        # Sort chronologically
        df = df.sort_values(["year", "quarter_int"]).reset_index(drop=True)

        # --- Feature Engineering Step ---
        df["revenue_log"] = _safe_log1p(df["revenue"])
        df["revenue_log_lag1"] = df["revenue_log"].shift(1)
        df["revenue_log_lag4"] = df["revenue_log"].shift(4)

        if {"operatingIncome", "revenue"}.issubset(df.columns):
            df["operatingIncomeRatio"] = _safe_div(df["operatingIncome"], df["revenue"])
        if {"grossProfit", "revenue"}.issubset(df.columns):
            df["grossProfitRatio"] = _safe_div(df["grossProfit"], df["revenue"])
        if {"netIncome", "revenue"}.issubset(df.columns):
            df["netIncomeRatio"] = _safe_div(df["netIncome"], df["revenue"])

        # YoY Lags
        for col in ["revenue_log", "operatingIncome", "netIncome"]:
            if col in df.columns:
                df[f"{col}_yoy"] = _safe_div(df[col], df[col].shift(4)) - 1.0

        # Asset/Equity Log Lags (Critical for TFT Known Future Context)
        if "totalAssets" in df.columns:
            df["totalAssets_lag1"] = df["totalAssets"].shift(1)
            df["totalAssets_lag1_log"] = _safe_log1p(df["totalAssets_lag1"])
        if "totalEquity" in df.columns:
            df["totalEquity_lag1"] = df["totalEquity"].shift(1)
            df["totalEquity_lag1_log"] = _safe_log1p(df["totalEquity_lag1"])

        # Ensure gics_sectors exists (Fallback to FMP sector)
        df["gics_sectors"] = df.get("sector", pd.NA)

        # Save to the same directory with the _feature suffix
        out_path = os.path.join(DATA_DIR, f"{ticker}_feature.csv")
        df.to_csv(out_path, index=False)
        print(f"Processed & Engineered {ticker}: {len(df)} quarters -> {out_path}")
        processed_count += 1
        
    if processed_count == 0:
        print("\n[ERROR] No files were processed. Did you run fetch_megacap5.py first?")

if __name__ == "__main__":
    main()