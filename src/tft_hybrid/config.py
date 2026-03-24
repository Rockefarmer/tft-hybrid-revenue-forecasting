from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    financial_data_dir: str = r"C:\ThesisResearch\finbert_tft\data\financial_data"
    finbert_quarterly_csv: str = r"C:\ThesisResearch\finbert_tft\data\processed_sentiment\megacap5_finbert_quarterly.csv"
    llama3_quarterly_csv: str = r"C:\ThesisResearch\finbert_tft\data\processed_sentiment\megacap5_quarterly_sentiment_llama3.csv"
    transcripts_dir: str = r"C:\ThesisResearch\finbert_tft\data\earning_scripts"
    processed_sentiment_dir: str = r"C:\ThesisResearch\finbert_tft\data\processed_sentiment"
    finbert_out_dir: str = r"C:\ThesisResearch\finbert_tft\results\finbert_TFT_MegaCap5_h4"
    llama3_out_dir: str = r"C:\ThesisResearch\finbert_tft\results\llama3_TFT_MegaCap5_h4"


MEGA_CAP_5 = ["AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "NVDA"]
SPLIT_TRAIN = ("2007Q1", "2019Q4")
SPLIT_VAL = ("2020Q1", "2022Q3")
SPLIT_TEST = ("2022Q4", "2025Q2")
SEED = 42
