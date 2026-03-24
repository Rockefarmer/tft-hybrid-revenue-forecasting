# -*- coding: utf-8 -*-
"""
STEP 4: FinBERT Transcript Preprocessing Pipeline
=================================================
Thesis Methodology: Multimodal Forecasting
Isolated Goal: Transform raw .txt earnings call transcripts into a clean, structured
               dataset of Forward-Looking Managerial Sentences optimized for FinBERT inference.

Implementation Details:
1. Filters for Mega-Cap 5 companies.
2. Extracts metadata (Ticker, YQ) from filename.
3. Cleans noise (boileplate, operator chat, transcriber artifacts).
4. Isolates Managerial speech (discards analysts).
5. Segments Prepared Remarks vs. Q&A (prioritizing Remarks).
6. Segments sentences (BERT-compliant).
7. Filters for FLI (Forward-Looking Indicators) only.
"""

import os
import re
import glob
import warnings
import pandas as pd
from datetime import datetime

# Handle potential issues with text encoding
warnings.filterwarnings("ignore", category=UserWarning)

# Attempt to load NLTK for robust sentence splitting.
# If not available, fallback to basic regex (though NLP library is highly recommended).
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Download the punkt tokenizer model if not present (only needs to run once)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("[INFO] NLTK Punkt model not found. Downloading...")
        nltk.download('punkt')
        print("[INFO] Download complete.")
    USE_NLTK = True
except ImportError:
    print("[WARN] NLTK is not installed. Falling back to basic Regex sentence splitting.")
    USE_NLTK = False

# ------------------ USER CONFIG ------------------
# Define your specific MegaCap 5 cohort. GOOGL handles GOOGL/GOOG files.
MEGA_CAP_5 = ["AAPL", "AMZN", "GOOGL", "GOOG", "MSFT", "NVDA"]

# Root directory for raw text transcripts
INPUT_ROOT_DIR = r"C:\ThesisResearch\finbert_tft\data\earning_scripts"

# Output directory for processed dataset
OUTPUT_DIR = r"C:\ThesisResearch\finbert_tft\data\processed_sentiment"

# Output filename for the finalized aggregate dataset
OUTPUT_FILENAME = "megacap5_forward_looking_sentences.csv"

# Minimum sentence length in characters to keep (filters short chatter)
MIN_SENT_LEN = 20

# dictionary of FLIs (Forward-Looking Indicators)
FLI_KEYWORDS = [
    r'anticipate', r'believe', r'commit', r'continue', r'could', r'estimate',
    r'expect', r'forecast', r'goal', r'guidance', r'intend', r'may', r'might',
    r'objective', r'outlook', r'plan', r'predict', r'project', r'seek', r'should',
    r'strategy', r'target', r'will', r'would', r'next quarter', r'next year',
    r'upcoming', r'foresee', r'projection', r'expected'
]

# Speaker/role heuristics for executive-only filtering
EXEC_TITLE_KEYWORDS = [
    r'chief executive officer', r'ceo', r'chief financial officer', r'cfo',
    r'chief operating officer', r'coo', r'president', r'chair', r'chairman',
    r'chairwoman', r'senior vice president', r'vice president', r'vp',
    r'treasurer', r'investor relations', r'general counsel', r'chief', r'officer'
]

EXCLUDE_SPEAKER_KEYWORDS = [
    r'operator', r'analyst', r'question', r'q&a', r'moderator'
]

QNA_KEYWORDS = [
    r'question-and-answer session', r'question and answer session', r'questions and answers',
    r'we will now take questions', r'we will now open the line for questions',
    r'open the line for questions', r'open to questions', r'ready for questions',
    r'let\'?s turn to q&a', r'let\'?s go to q&a', r'q\s*&\s*a', r'q\s*and\s*a'
]

SAFE_HARBOR_START = [
    r'safe harbor', r'forward-looking statements', r'cautionary statement',
    r'private securities litigation reform act'
]

SAFE_HARBOR_END = [
    r'form 10-k', r'form 10-q', r'form 8-k', r'sec filings',
    r'assumes no obligation', r'no obligation to update', r'turn the call over'
]

# Use transcript header date to align all outputs to calendar quarter.
# If date cannot be extracted, fallback to filename-derived quarter.
USE_CALENDAR_QUARTER = True

NOISE_PATTERNS = [
    r'you may begin your conference',
    r'my name is .* conference operator',
    r'at this time, i would like to welcome everyone',
    r'after the speakers\s*[\'’]?\s*(remarks|presentation),? there will be',
    r'after the speaker\s*[\'’]?\s*(remarks|presentation),? there will be',
    r'we will now open (the call )?for questions',
    r'now we will open up the call for questions',
    r'operator,? would you please poll for questions',
    r'webcast will be available for replay',
    r'during this call, we may make',
    r'during this call, we will discuss non-gaap',
    r'let me turn the call over',
    r'in closing, let me highlight upcoming events for the financial community',
]
# -------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ HELPER FUNCTIONS ------------------

def _parse_filename(filepath):
    """
    Parses metadata (Ticker, YQ) from standard standard filenames.
    Assumes structure: root/scripts/TICKER_YEAR_QUARTER.txt or root/scripts/TICKER/YEAR/QUARTER.txt
    """
    fname = os.path.basename(filepath)
    ticker = None
    yq_key = None
    
    # Matching pattern: TICKER_YYYY_QX.txt (Case Insensitive)
    match = re.search(r'([A-Z]+)_(\d{4})_(Q[1-4])', fname, re.IGNORECASE)
    
    if match:
        ticker = match.group(1).upper()
        # Consolidate GOOG and GOOGL under GOOGL
        if ticker == "GOOG": ticker = "GOOGL"
        
        year = match.group(2)
        quarter = match.group(3).upper()
        yq_key = f"{year}{quarter}"
        
    return ticker, yq_key


def _clean_cross_talk(text):
    """Removes transcriber artifacts like [Applause] or (Inaudible)"""
    # Remove brackets [...]
    text = re.sub(r'\[.*?\]', '', text)
    # Remove parentheses (...)
    text = re.sub(r'\(.*?\)', '', text)
    return text.strip()


def _remove_safe_harbor_blocks(text):
    """Aggressively removes Safe Harbor disclaimers that bias sentiment."""
    lowered = text.lower()
    start_idx = -1
    for kw in SAFE_HARBOR_START:
        match = re.search(kw, lowered)
        if match:
            start_idx = match.start()
            break

    if start_idx == -1:
        return text

    end_idx = -1
    for kw in SAFE_HARBOR_END:
        match = re.search(kw, lowered[start_idx:])
        if match:
            end_idx = start_idx + match.end()
            break

    if end_idx == -1:
        # If no clear end marker, remove a conservative window after the start.
        end_idx = min(len(text), start_idx + 2000)

    return (text[:start_idx] + "\n" + text[end_idx:]).strip()


def _is_safe_harbor_boilerplate(sentence):
    """Detects standard forward-looking disclaimer sentences."""
    safe_keywords = [
        r'safe harbor', r'factors that could cause', r'actual results may differ',
        r'sec filings', r'forward-looking statements', r'no obligation to update'
    ]
    text_lower = sentence.lower()
    return any(re.search(kw, text_lower) for kw in safe_keywords)


def _detect_qna_transition(text):
    """Locates standard transitions statements from Presentation to Q&A."""
    text_lower = text.lower()
    min_position = int(len(text_lower) * 0.35)
    for kw in QNA_KEYWORDS:
        for match in re.finditer(kw, text_lower):
            # Ignore early operator/scripted intro mentions.
            if match.start() >= min_position:
                return match.start()
    return -1


def _is_fli_sentence(sentence):
    """Filters sentences containing forward-looking indicators (FLIs)."""
    text_lower = sentence.lower()
    return any(re.search(kw, text_lower) for kw in FLI_KEYWORDS)


def _split_sentences(text):
    """Segments a block of speech into coherent sentences."""
    if USE_NLTK:
        # Preferred: Use NLP library
        return sent_tokenize(text)
    else:
        # Fallback: Basic regex split on .!? (less accurate, handles abbreviations poorly)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def _extract_header_date(raw_text):
    """Extract a transcript date from header text (e.g., 'Aug. 24, 2022')."""
    match = re.search(
        r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|'
        r'Nov(?:ember)?|Dec(?:ember)?)\.?\s+(\d{1,2}),\s+(\d{4})\b',
        raw_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    month_name = match.group(1).lower()[:3]
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = month_map.get(month_name)
    day = int(match.group(2))
    year = int(match.group(3))
    if not month:
        return None
    return datetime(year, month, day)


def _resolve_year_quarter(raw_text, ticker, fallback_yq_key):
    """Resolve output Year_Quarter; optionally override for fiscal/calendar mismatches."""
    if not USE_CALENDAR_QUARTER:
        return fallback_yq_key

    dt = _extract_header_date(raw_text)
    if not dt:
        return fallback_yq_key

    quarter = ((dt.month - 1) // 3) + 1
    return f"{dt.year}Q{quarter}"


def _is_noise_sentence(sentence):
    text_lower = sentence.lower().strip()
    return any(re.search(pattern, text_lower) for pattern in NOISE_PATTERNS)


def _normalize_name(name):
    return re.sub(r'\s+', ' ', name.strip()).upper()


def _extract_executives(raw_text):
    """Extract executive names from the transcript header when present."""
    executives = set()
    block_patterns = [
        (r'Executives\s*(.*?)\s*Analysts', re.IGNORECASE | re.DOTALL),
        (r'Company Participants\s*(.*?)\s*Conference Call Participants', re.IGNORECASE | re.DOTALL),
        (r'Company Participants\s*(.*?)\s*Analysts', re.IGNORECASE | re.DOTALL),
        (r'Company Participants\s*(.*?)\s*Operator', re.IGNORECASE | re.DOTALL),
    ]

    blocks = []
    for pattern, flags in block_patterns:
        match = re.search(pattern, raw_text, flags=flags)
        if match:
            blocks.append(match.group(1))

    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: Name - Title (includes Unicode en dash variants)
            name = re.split(r'\s+[\-\u2013]\s+', line)[0]
            if name:
                executives.add(_normalize_name(name))

    return executives


def _is_executive_speaker(header, executives):
    """Decide if a speaker header belongs to management."""
    header_norm = _normalize_name(header)
    if header_norm in executives:
        return True

    header_lower = header.lower()
    if any(re.search(kw, header_lower) for kw in EXCLUDE_SPEAKER_KEYWORDS):
        return False

    return any(re.search(kw, header_lower) for kw in EXEC_TITLE_KEYWORDS)


def _iter_speaker_blocks(text, executives):
    """Yield (speaker, speech) pairs using robust header detection."""
    speaker = None
    buffer = []

    for line in text.splitlines():
        stripped = line.strip()

        # Detect potential speaker header lines
        is_header = False
        header_text = stripped
        if not stripped:
            if speaker and buffer:
                buffer.append('')
            continue

        # Pattern: Name - Role / Name: (includes Unicode en dash variants)
        if re.match(r"^[A-Za-z][A-Za-z\s'\.\-]+\s*[\-\u2013:].*", stripped):
            is_header = True
        # Pattern: Name only (common in SA transcripts)
        elif _normalize_name(stripped) in executives and len(stripped.split()) <= 4:
            is_header = True

        if is_header:
            if speaker and buffer:
                yield speaker, " ".join(buffer).strip()
            speaker = header_text
            buffer = []
            continue

        if speaker:
            buffer.append(stripped)

    if speaker and buffer:
        yield speaker, " ".join(buffer).strip()


# ------------------ CORE PARSING LOGIC ------------------

def parse_transcript(filepath, ticker, yq_key):
    """
    Reads a raw text transcript and executes the preprocessing pipeline:
    Cleaning -> Segmenting (Managers vs Analysts) -> Segmenting (Sents) -> Filtering (FLIs).
    """
    processed_sentences = []
    
    print(f"[INFO] Processing: {ticker} | {yq_key}")
    
    try:
        # Earnings calls are typically ASCII or UTF-8. Using utf-8 with fallback.
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            raw_text = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")
        return []

    resolved_yq_key = _resolve_year_quarter(raw_text, ticker, yq_key)

    # 1. Cleaning Phase 1 (transcriber artifacts + safe harbor)
    cleaned_text = _clean_cross_talk(raw_text)
    cleaned_text = _remove_safe_harbor_blocks(cleaned_text)

    # 2. Segmenting Phase 1: Section Isolation (Prepared Remarks vs Q&A)
    # We prioritize the Prepared Remarks section for forward-looking guidance.
    qna_start = _detect_qna_transition(cleaned_text)
    presentation_text = cleaned_text
    
    if qna_start != -1:
        presentation_text = cleaned_text[:qna_start]
        # print(f" - Q&A start found. Presentation section length: {len(presentation_text)} chars.")
    else:
        # Fallback: If no explicit Q&A transition found, assume first significant block is Executive Prepared Remarks
        # until the first logical change, or process the whole file, relying heavily on speaker identification.
        # Sticking to Q&A split is safer for data quality.
        pass

    # 3. Segmenting Phase 2: Speaker Isolation (Managers only)
    executives = _extract_executives(raw_text)

    executive_narrative_blocks = []
    for header, speech_content in _iter_speaker_blocks(presentation_text, executives):
        if not _is_executive_speaker(header, executives):
            continue
        if len(speech_content) > 50:
            executive_narrative_blocks.append(speech_content)

    if not executive_narrative_blocks:
        print(f"[WARN] No executive speaker blocks detected in {ticker}_{yq_key}. Skipping.")
        return []

    # Combine isolated executive narrative
    combined_executive_narrative = " ".join(executive_narrative_blocks)

    if not combined_executive_narrative or len(combined_executive_narrative) < 100:
        print(f"[WARN] Isolated executive narrative is too short (< 100 chars) for {ticker}_{yq_key}. Skipping.")
        return []

    # 4. Segmenting Phase 3: Sentence Segmentation (BERT-compliant)
    all_sentences = _split_sentences(combined_executive_narrative)

    # 5. Filtering Phase: FLI Isolation (Anticipating structural breaks)
    for sent in all_sentences:
        clean_sent = sent.strip()
        
        # Validation checks
        if len(clean_sent) < MIN_SENT_LEN: continue
        if clean_sent.count(' ') < 3: continue # Filter single-word confirmations
        if _is_safe_harbor_boilerplate(clean_sent): continue # Filter disclaimers
        if _is_noise_sentence(clean_sent): continue
        
        # Keep ONLY Forward-Looking Statements
        if _is_fli_sentence(clean_sent):
            processed_sentences.append({
                'ticker': ticker,
                'Year_Quarter': resolved_yq_key,
                'sentence': clean_sent
            })

    # print(f" - FLI Filtering complete. Kept {len(processed_sentences)} out of {len(all_sentences)} total executive sentences.")
    
    return processed_sentences


# ------------------ MAIN LOOP ------------------

def main():
    print("=========================================================")
    print("STEP 4: FinBERT Transcript Preprocessing Pipeline Started")
    print(f"Cohort: {MEGA_CAP_5}")
    print(f"Sentence Segmenter: {'NLTK' if USE_NLTK else 'Regex Fallback'}")
    print("=========================================================")

    all_processed_data = []
    files_processed_count = 0

    # recursively find all .txt files in the input folder
    search_pattern = os.path.join(INPUT_ROOT_DIR, '**', '*.txt')
    target_files = glob.glob(search_pattern, recursive=True)

    if not target_files:
        raise RuntimeError(f"No .txt files found in {INPUT_ROOT_DIR}")

    print(f"[INFO] Found {len(target_files)} total .txt files. Filtering for Mega-Cap 5...")

    for filepath in target_files:
        
        # Apply Mega-Cap 5 Filtering
        fname = os.path.basename(filepath).upper()
        if not any(f"{ticker}" in fname for ticker in MEGA_CAP_5):
            continue 

        # Extract metadata
        ticker, yq_key = _parse_filename(filepath)
        if not ticker or not yq_key:
            print(f"[WARN] Skipping {filepath} - Standard filename TICKER_YYYY_QX.txt not detected.")
            continue

        # run the parsing pipeline
        file_sentences = parse_transcript(filepath, ticker, yq_key)
        
        if file_sentences:
            all_processed_data.extend(file_sentences)
            files_processed_count += 1

    print("=========================================================")
    print("Preprocessing Loop Complete.")
    print(f"Files processed for MegaCap 5: {files_processed_count}")
    print(f"Total FLI sentences isolated: {len(all_processed_data)}")

    # Convert to DataFrame and save
    if all_processed_data:
        processed_df = pd.DataFrame(all_processed_data)
        
        # Ensure correct column ordering
        processed_df = processed_df[['ticker', 'Year_Quarter', 'sentence']]
        
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        # Use sep='\t' if sentences contain many commas (standard in conversational text)
        processed_df.to_csv(output_path, index=False, encoding='utf-8', sep='\t')
        
        print(f"[SUCCESS] Finalized dataset saved to {output_path}")
        print("Final DataFrame structure:")
        print(processed_df.head(5))
    else:
        print("[ERROR] No forward-looking data isolated. Final dataset was not saved.")

if __name__ == "__main__":
    main()