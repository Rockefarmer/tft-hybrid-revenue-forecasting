import os
import pandas as pd


def canonical_ticker(ticker: str) -> str:
    t = str(ticker).upper().strip()
    return "GOOGL" if t == "GOOG" else t


def merge_finbert_features(data: pd.DataFrame, finbert_csv: str) -> pd.DataFrame:
    sent = pd.read_csv(finbert_csv)
    required = {"ticker", "Year_Quarter", "net_sentiment", "finbert_pos", "finbert_neg", "finbert_neu"}
    missing = required - set(sent.columns)
    if missing:
        raise ValueError(f"FinBERT CSV missing required columns: {sorted(missing)}")
    sent = sent[list(required)].copy()
    sent["ticker"] = sent["ticker"].astype(str).map(canonical_ticker)
    sent["Year_Quarter"] = sent["Year_Quarter"].astype(str).str.upper().str.strip()
    sent = sent.groupby(["ticker", "Year_Quarter"], as_index=False).mean(numeric_only=True)
    data = data.merge(sent, on=["ticker", "Year_Quarter"], how="left")
    for col in ["net_sentiment", "finbert_pos", "finbert_neg", "finbert_neu"]:
        out_col = f"sent_{col}"
        data[out_col] = data.groupby("ticker")[col].shift(1)
        data[f"{out_col}_ffill"] = data.groupby("ticker")[out_col].ffill()
        data[out_col] = data[out_col].fillna(0.0)
        data[f"{out_col}_ffill"] = data[f"{out_col}_ffill"].fillna(0.0)
    return data.drop(columns=["net_sentiment", "finbert_pos", "finbert_neg", "finbert_neu"], errors="ignore")


def merge_llama3_features(data: pd.DataFrame, llama3_csv: str) -> pd.DataFrame:
    sent = pd.read_csv(llama3_csv)
    required = {"ticker", "Year_Quarter", "sent_net_mean", "sent_fli_count"}
    missing = required - set(sent.columns)
    if missing:
        raise ValueError(f"Llama-3 CSV missing required columns: {sorted(missing)}")
    sent = sent[list(required)].copy()
    sent["ticker"] = sent["ticker"].astype(str).map(canonical_ticker)
    sent["Year_Quarter"] = sent["Year_Quarter"].astype(str).str.upper().str.strip()
    sent = sent.groupby(["ticker", "Year_Quarter"], as_index=False).mean(numeric_only=True)
    data = data.merge(sent, on=["ticker", "Year_Quarter"], how="left")
    for col in ["sent_net_mean", "sent_fli_count"]:
        data[col] = data.groupby("ticker")[col].shift(1)
        data[f"{col}_ffill"] = data.groupby("ticker")[col].ffill()
        data[col] = data[col].fillna(0.0)
        data[f"{col}_ffill"] = data[f"{col}_ffill"].fillna(0.0)
    return data
