# Repo Plan

## Main branches of the study

### A. Baseline TFT
- Input: structured financial data only
- Output: revenue forecasts for h=1 and h=1..4

### B. FinBERT + TFT
- Input: structured data + FinBERT-derived sentiment/tone features
- Output: same horizons for direct comparison with baseline

### C. Llama-3 + TFT
- Input: structured data + Llama-3-derived narrative features
- Output: same horizons for direct comparison with baseline and FinBERT hybrid

## Suggested implementation order

1. Data ingestion and panel cleaning
2. Structured feature engineering
3. Chronological splitting
4. Baseline TFT training
5. Multi-horizon extension
6. FinBERT feature extraction and alignment
7. Llama-3 feature extraction and alignment
8. Evaluation, plots, and ablations
