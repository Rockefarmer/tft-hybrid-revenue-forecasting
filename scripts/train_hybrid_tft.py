import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# ==========================================
# Bypass PyTorch 2.6 security for smooth loading
# ==========================================
_original_load = torch.load
def _trusted_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _trusted_load
# ==========================================

def build_and_train_hybrid_tft(data_path: str):
    print("Loading Hybrid Multimodal Dataset...")
    df = pd.read_csv(data_path)
    
    df["time_idx"] = df["time_idx"].astype(int)
    df["ticker"] = df["ticker"].astype(str)
    
    # 1. Target Variable setup
    df["revenue_log"] = np.log1p(df["revenue"])
    df["revenue_log_lag1"] = df.groupby("ticker")["revenue_log"].shift(1)
    
    # Drop rows with NaN in critical lagged features (first quarter of each ticker)
    df = df.dropna(subset=["revenue_log_lag1", "net_sentiment_lag1"]).reset_index(drop=True)

    # 2. Strict Macro-Economic Period Splitting (Same as Chapter 4 & 5)
    max_time_idx = df["time_idx"].max()
    val_cutoff = int(df[df['Year_Quarter'] == '2020Q4']['time_idx'].iloc[0])
    test_cutoff = int(df[df['Year_Quarter'] == '2022Q4']['time_idx'].iloc[0])
    
    print(f"Training ends at 2020Q4 (time_idx: {val_cutoff})")
    print(f"Test Set covers 2023Q1 to 2025 (AI Boom Structural Break)")

    # 3. Defining the Hybrid Feature Set
    # Mixing the best quantitative features with our new NLP features
    potential_features = [
        "revenue_log_lag1", "totalAssets", "rnd", "netIncome", "grossProfit",
        "net_sentiment_lag1", "finbert_pos_lag1", "finbert_neg_lag1"  # <-- The NLP Magic
    ]
    actual_features = [col for col in potential_features if col in df.columns]
    print(f"Injecting Features into the Network: {actual_features}")
    
    max_prediction_length = 4  
    max_encoder_length = 8 # Shorter encoder to prevent overfitting the small sample

    train_df = df[df.time_idx <= val_cutoff]

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="revenue_log", 
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length // 2, 
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        
        static_categoricals=["ticker"],
        time_varying_known_reals=["time_idx", "Q1", "Q2", "Q3", "Q4"],
        time_varying_unknown_reals=actual_features,
        
        target_normalizer=GroupNormalizer(
            groups=["ticker"], transformation="softplus", center=False
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    val_df = df[df.time_idx <= test_cutoff]
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=True, stop_randomization=True
    )
    
    testing = TimeSeriesDataSet.from_dataset(
        training, df, min_prediction_idx=test_cutoff + 1, predict=False, stop_randomization=True
    )

    batch_size = 16
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)
    test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    # 4. Initialize the LIGHTWEIGHT TFT
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,              # Kept small to avoid the Curse of Dimensionality
        lstm_layers=1,               # Single layer
        attention_head_size=2,       
        dropout=0.15,
        hidden_continuous_size=8,   
        loss=QuantileLoss(),     
        reduce_on_plateau_patience=4,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=15, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()
    
    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="auto", 
        enable_model_summary=False,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
    )

    return trainer, tft, train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    DATA_PATH = r"C:\ThesisResearch\finbert_tft\data\tft_hybrid_panel.csv"
    
    print("Initializing Hybrid FinBERT-TFT Model...")
    trainer, tft_model, train_dl, val_dl, test_dl = build_and_train_hybrid_tft(DATA_PATH)
    
    print("\n--- Starting Training ---")
    trainer.fit(
        tft_model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"\nBest model saved at: {best_model_path}")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    print("\n--- Evaluation on 2023-2025 Test Set ---")
    best_tft.eval()
    device = next(best_tft.parameters()).device

    actuals_log = torch.cat([y[0] for x, y in iter(test_dl)]).to(device)
    predictions_log = best_tft.predict(test_dl).to(device)

    actuals_level = torch.expm1(actuals_log)
    predictions_level = torch.expm1(predictions_log)

    overall_mae = F.l1_loss(predictions_level, actuals_level)
    overall_rmse = torch.sqrt(F.mse_loss(predictions_level, actuals_level))
    overall_mape = torch.mean(torch.abs((predictions_level - actuals_level) / actuals_level))

    print(f"\n[Hybrid Model - Overall Metrics]")
    print(f"Overall MAE   : {overall_mae.item():.4f}")
    print(f"Overall RMSE  : {overall_rmse.item():.4f}")
    print(f"Overall MAPE  : {overall_mape.item():.4f}")

    print("\n[Hybrid Model - Metrics by Forecast Horizon]")
    for i in range(4):
        pred_h = predictions_level[:, i]
        act_h = actuals_level[:, i]
        mape_h = torch.mean(torch.abs((pred_h - act_h) / act_h))
        print(f"Horizon t+{i+1} (Quarter {i+1}) MAPE : {mape_h.item():.4f}")