# tft-hybrid-revenue-forecasting

Hybrid multimodal revenue forecasting combining TFT with FinBERT sentiment and Llama-3 narrative features.

This repo refactors the original hybrid research scripts into a cleaner structure with two experiment tracks:

- **FinBERT + TFT** for quarterly sentiment features
- **Llama-3 + TFT** for forward-looking narrative indicators

It keeps the same thesis-style workflow reflected in your original files: fetch Mega-Cap 5 fundamentals, engineer structured quarterly features, preprocess transcripts, merge text-derived signals, and train fixed-split multi-horizon TFT models.

## Refactor mapping

- `fetch_megacap5.py` → `src/tft_hybrid/fetch.py`
- `preprocess_features_megacap5.py` → `src/tft_hybrid/structured_features.py`
- `finbert_preprocess.py` → `src/tft_hybrid/transcripts.py`
- `finbert_tft.py` / `llama3_tft_h4.py` → `src/tft_hybrid/trainers.py`
- `tft_megacap5_h4_metrics.py` → `src/tft_hybrid/trainers.py` (baseline-compatible path)
- `train_hybrid_tft.py` → preserved as a lightweight alternative training prototype in `docs/notes.md`

## Proposed layout

```text
tft-hybrid-revenue-forecasting/
├── configs/
│   ├── finbert_h4.yaml
│   └── llama3_h4.yaml
├── data/
├── docs/
├── notebooks/
├── reports/
├── scripts/
│   ├── fetch_megacap5.py
│   ├── preprocess_structured_features.py
│   ├── preprocess_finbert_transcripts.py
│   ├── train_finbert_h4.py
│   └── train_llama3_h4.py
├── src/
│   └── tft_hybrid/
│       ├── config.py
│       ├── fetch.py
│       ├── time_utils.py
│       ├── structured_features.py
│       ├── transcripts.py
│       ├── merge_text_features.py
│       ├── dataset.py
│       ├── evaluate.py
│       └── trainers.py
├── tests/
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_megacap5.py
python scripts/preprocess_structured_features.py
python scripts/preprocess_finbert_transcripts.py
python scripts/train_finbert_h4.py
# or
python scripts/train_llama3_h4.py
```

Update the Windows paths in `src/tft_hybrid/config.py` before running.
