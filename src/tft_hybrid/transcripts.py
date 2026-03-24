import glob
import os
import re
import warnings
from datetime import datetime

import pandas as pd

from .config import Paths, MEGA_CAP_5

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    USE_NLTK = True
except ImportError:
    USE_NLTK = False

MIN_SENT_LEN = 20
FLI_KEYWORDS = [
    r"anticipate", r"believe", r"commit", r"continue", r"could", r"estimate",
    r"expect", r"forecast", r"goal", r"guidance", r"intend", r"may", r"might",
    r"objective", r"outlook", r"plan", r"predict", r"project", r"seek", r"should",
    r"strategy", r"target", r"will", r"would", r"next quarter", r"next year",
    r"upcoming", r"foresee", r"projection", r"expected",
]
EXEC_TITLE_KEYWORDS = [
    r"chief executive officer", r"ceo", r"chief financial officer", r"cfo",
    r"chief operating officer", r"coo", r"president", r"chair", r"senior vice president",
    r"vice president", r"vp", r"treasurer", r"investor relations", r"general counsel",
]
EXCLUDE_SPEAKER_KEYWORDS = [r"operator", r"analyst", r"question", r"q&a", r"moderator"]


def _parse_filename(filepath):
    fname = os.path.basename(filepath)
    match = re.search(r"([A-Z]+)_(\d{4})_(Q[1-4])", fname, re.IGNORECASE)
    if not match:
        return None, None
    ticker = match.group(1).upper()
    if ticker == "GOOG":
        ticker = "GOOGL"
    return ticker, f"{match.group(2)}{match.group(3).upper()}"


def _split_sentences(text: str):
    if USE_NLTK:
        return sent_tokenize(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _is_fli_sentence(sentence: str) -> bool:
    lower = sentence.lower()
    return any(re.search(kw, lower) for kw in FLI_KEYWORDS)


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).upper()


def _extract_executives(raw_text: str):
    executives = set()
    patterns = [
        (r"Executives\s*(.*?)\s*Analysts", re.IGNORECASE | re.DOTALL),
        (r"Company Participants\s*(.*?)\s*Conference Call Participants", re.IGNORECASE | re.DOTALL),
        (r"Company Participants\s*(.*?)\s*Analysts", re.IGNORECASE | re.DOTALL),
    ]
    for pattern, flags in patterns:
        match = re.search(pattern, raw_text, flags=flags)
        if not match:
            continue
        for line in match.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            name = re.split(r"\s+[\-–]\s+", line)[0]
            if name:
                executives.add(_normalize_name(name))
    return executives


def _is_executive_speaker(header: str, executives) -> bool:
    norm = _normalize_name(header)
    if norm in executives:
        return True
    lower = header.lower()
    if any(re.search(kw, lower) for kw in EXCLUDE_SPEAKER_KEYWORDS):
        return False
    return any(re.search(kw, lower) for kw in EXEC_TITLE_KEYWORDS)


def _iter_speaker_blocks(text: str, executives):
    speaker = None
    buffer = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_header = bool(re.match(r"^[A-Za-z][A-Za-z\s'\.\-]+\s*[\-–:].*", stripped)) or (_normalize_name(stripped) in executives and len(stripped.split()) <= 4)
        if is_header:
            if speaker and buffer:
                yield speaker, " ".join(buffer).strip()
            speaker = stripped
            buffer = []
            continue
        if speaker:
            buffer.append(stripped)
    if speaker and buffer:
        yield speaker, " ".join(buffer).strip()


def parse_transcript(filepath, ticker, yq_key):
    with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
        raw_text = handle.read()
    executives = _extract_executives(raw_text)
    sentences = []
    for header, speech in _iter_speaker_blocks(raw_text, executives):
        if not _is_executive_speaker(header, executives):
            continue
        for sent in _split_sentences(speech):
            clean = sent.strip()
            if len(clean) < MIN_SENT_LEN:
                continue
            if _is_fli_sentence(clean):
                sentences.append({"ticker": ticker, "Year_Quarter": yq_key, "sentence": clean})
    return sentences


def main():
    paths = Paths()
    os.makedirs(paths.processed_sentiment_dir, exist_ok=True)
    search_pattern = os.path.join(paths.transcripts_dir, "**", "*.txt")
    target_files = glob.glob(search_pattern, recursive=True)
    all_rows = []
    for filepath in target_files:
        fname = os.path.basename(filepath).upper()
        if not any(f"{ticker}" in fname for ticker in MEGA_CAP_5):
            continue
        ticker, yq_key = _parse_filename(filepath)
        if not ticker or not yq_key:
            continue
        all_rows.extend(parse_transcript(filepath, ticker, yq_key))
    if not all_rows:
        print("[ERROR] No forward-looking sentences isolated.")
        return
    out_df = pd.DataFrame(all_rows)[["ticker", "Year_Quarter", "sentence"]]
    out_path = os.path.join(paths.processed_sentiment_dir, "megacap5_forward_looking_sentences.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8", sep="	")
    print(f"Saved {len(out_df)} rows -> {out_path}")
