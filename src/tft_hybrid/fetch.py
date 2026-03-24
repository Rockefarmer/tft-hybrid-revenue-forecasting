import os
import pandas as pd
import requests
from tqdm import tqdm

from .config import Paths, MEGA_CAP_5


API_KEY = os.getenv("FMP_API_KEY")
BASE_V3 = "https://financialmodelingprep.com/api/v3"
START_DATE = "2007-01-01"
END_DATE = "2025-10-10"


session = requests.Session()


def _json_get(url: str):
    try:
        response = session.get(url, timeout=60)
        if response.status_code == 200:
            return response.json()
    except Exception as exc:
        print(f"Error GET {url}: {exc}")
    return None


def _within_window(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    mask = (df[date_col] >= pd.to_datetime(START_DATE)) & (df[date_col] <= pd.to_datetime(END_DATE))
    return df.loc[mask].sort_values(date_col).reset_index(drop=True)


def fetch_income_data(ticker: str) -> pd.DataFrame:
    url = f"{BASE_V3}/income-statement/{ticker}?period=quarter&limit=200&apikey={API_KEY}"
    js = _json_get(url)
    if not js:
        return pd.DataFrame()
    return _within_window(pd.DataFrame(js), "date")


def fetch_balance_sheet_data(ticker: str) -> pd.DataFrame:
    url = f"{BASE_V3}/balance-sheet-statement/{ticker}?period=quarter&limit=200&apikey={API_KEY}"
    js = _json_get(url)
    if not js:
        return pd.DataFrame()
    df = pd.DataFrame(js)
    if df.empty:
        return df
    df = _within_window(df, "date")
    equity_col = "totalEquity" if "totalEquity" in df.columns else "totalStockholdersEquity"
    keep = ["date"]
    if "totalAssets" in df.columns:
        keep.append("totalAssets")
    if equity_col in df.columns:
        keep.append(equity_col)
    df = df[keep].copy()
    if equity_col != "totalEquity" and equity_col in df.columns:
        df = df.rename(columns={equity_col: "totalEquity"})
    return df


def fetch_profile(ticker: str) -> dict:
    url = f"{BASE_V3}/profile/{ticker}?apikey={API_KEY}"
    js = _json_get(url)
    if js and isinstance(js, list) and len(js):
        row = js[0]
        return {"sector": row.get("sector"), "industry": row.get("industry")}
    return {"sector": None, "industry": None}


def main():
    if not API_KEY:
        raise RuntimeError("FMP_API_KEY environment variable is not set.")
    paths = Paths()
    os.makedirs(paths.financial_data_dir, exist_ok=True)
    print(f"Fetching data for Mega-Cap 5: {MEGA_CAP_5}")
    for ticker in tqdm(MEGA_CAP_5, desc="Downloading..."):
        inc = fetch_income_data(ticker)
        if inc.empty:
            print(f"[WARN] No income data for {ticker}")
            continue
        bs = fetch_balance_sheet_data(ticker)
        profile = fetch_profile(ticker)
        enriched = inc if bs.empty else inc.merge(bs, on="date", how="left")
        if bs.empty:
            enriched["totalAssets"] = pd.NA
            enriched["totalEquity"] = pd.NA
        enriched["sector"] = profile.get("sector")
        enriched["industry"] = profile.get("industry")
        enriched["ticker"] = ticker
        out = os.path.join(paths.financial_data_dir, f"{ticker}.csv")
        enriched.to_csv(out, index=False)
        print(f"Saved {ticker} -> {len(enriched)} quarters")
