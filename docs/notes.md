# Notes

This refactor is based on the original hybrid scripts:
- FinBERT quarterly sentiment merged into TFT inputs
- Llama-3 quarterly sentiment / forward-looking indicators merged into TFT inputs
- Fixed train/validation/test ranges consistent with the thesis design
- Baseline-compatible exports for `tft_test_predictions.csv`, `tft_test_agg_by_ticker.csv`, and `tft_test_overall_stats.csv`

The smaller `train_hybrid_tft.py` prototype is conceptually useful, but the more thesis-aligned logic comes from `finbert_tft.py` and `llama3_tft_h4.py`.
