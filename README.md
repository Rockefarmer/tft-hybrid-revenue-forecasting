
    # TFT Hybrid Models for Multi-Horizon Revenue Forecasting

    A research repository for **hybrid multi-horizon forecasting** with **Temporal Fusion Transformer (TFT)** plus text-derived features from earnings call transcripts.

    ## Scope

    This repo is for the **hybrid model layer** built on top of the structured `h=4` baseline:

    - **FinBERT + TFT** for sentiment/tone-based transcript features
    - **Llama-3 + TFT** for richer forward-looking narrative features
    - shared evaluation against the same `h=1..4` forecasting protocol

    ## Research Goal

    Use this repo to test whether transcript-derived signals can help reduce the accuracy drop that usually appears when forecasting further into the future.

    ## Included Variants

    - `configs/finbert_h4.yaml`
    - `configs/llama3_h4.yaml`
    - `scripts/run_finbert.py`
    - `scripts/run_llama3.py`
    - `src/tft_multihorizon/nlp/finbert_features.py`
    - `src/tft_multihorizon/nlp/llama3_features.py`
    - `src/tft_multihorizon/features/text_feature_alignment.py`

    ## Suggested Dependency Logic

    Keep the structured baseline repo as the canonical source for:

    - panel construction
    - chronological splits
    - base TFT training loop
    - metrics and reporting conventions

    Then use this repo for the NLP-specific pipeline and hybrid experiments.

    ## Example Commands

    ```bash
    python scripts/prepare_data.py
    python scripts/run_finbert.py --config configs/finbert_h4.yaml
    python scripts/run_llama3.py --config configs/llama3_h4.yaml
    ```

    ## Suggested GitHub Repo Name

    `tft-hybrid-revenue-forecasting`

    ## Recommended Positioning

    - Repo 1: clean baseline and methodology backbone
    - Repo 2: experimental hybrid layer for thesis Chapter 5 and ablation work
