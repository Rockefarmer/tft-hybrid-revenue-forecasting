# -*- coding: utf-8 -*-
"""
FinBERT + TFT MegaCap5 full-year (h=4) multi-horizon model.
Uses quarterly FinBERT sentiment features as additional inputs.
Output format matches tft_baseline_h4 / tft_megacap5_h4_metrics.

ADDED: Automatic generation of finalized 'per_epoch_metrics.csv' (Standardized Epoch summary).
VISUALIZATION: Updated plots to use finalized standardization and X-axis='epoch'.

FIXED:
1. NameError: 'tft_model' is not defined in main interpretability block.
2. EarlyStopping patience increased to 20 to allow > 100 epoch run.
3. Preprocessing typo [np.inf, -inf] fixed to -np.inf.
4. Visualization file-locking/standardization issues fixed.
"""

import os
import re
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

# Plotting libraries (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    plt = None
    sns = None
    HAS_PLOTTING = False

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
# Need CSVLogger to read metrics back programmatically
from lightning.pytorch.loggers import CSVLogger 

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# Set seaborn style for prettier plots
if HAS_PLOTTING:
    sns.set_theme(style="whitegrid")

# for a tiny speed boost on your 4060
torch.set_float32_matmul_precision("high")

# ------------------ USER CONFIG ------------------
DATA_DIR  = r"C:\ThesisResearch\finbert_tft\data\financial_data"
FINBERT_CSV = r"C:\ThesisResearch\finbert_tft\data\processed_sentiment\megacap5_finbert_quarterly.csv"
OUT_DIR   = r"C:\ThesisResearch\finbert_tft\results\finber_TFT_MegaCap5_h4"

MEGA_CAP_5 = ["AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "NVDA"]

# Fixed splits (inclusive) - Aligned with experimental design
SPLIT_TRAIN = ("2007Q1", "2019Q4")
SPLIT_VAL   = ("2020Q1", "2022Q3")
SPLIT_TEST  = ("2022Q4", "2025Q2")

SEED            = 42
MAX_EPOCHS      = 150    # Use bigger epoch from 50 to 150
BATCH_SIZE      = 64
LR              = 1e-3
WEIGHT_DECAY    = 1e-5
HIDDEN_SIZE     = 64
ATTN_HEADS      = 4
DROPOUT         = 0.15
MAX_ENCODER_LEN = 12     # 12 quarters of historical context
MAX_PRED_LEN    = 4      # Full fiscal year (h=4)
# -------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ HELPERS ------------------
def _safe_log1p(x):
    return np.log1p(np.clip(x, a_min=0, a_max=None))

def _parse_yq_str(yq: str):
    s = str(yq).strip().upper()
    m = re.search(r'(\d{4})\s*[-/ ]?\s*Q\s*([1-4])', s)
    if not m:
        raise ValueError(f"Cannot parse quarter string: {yq!r}")
    return int(m.group(1)), int(m.group(2))

def _yq_scalar(y, q) -> int:
    return int(y) * 4 + (int(q) - 1)

def _yq_key_vec(year_series, quarter_series) -> pd.Series:
    y = pd.to_numeric(year_series, errors="coerce")
    q = pd.to_numeric(quarter_series, errors="coerce")
    return (y * 4 + (q - 1)).astype("Int64")

def _ensure_year_quarter(df: pd.DataFrame):
    if "year" not in df.columns:
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            df["year"] = dt.dt.year
        elif "Year_Quarter" in df.columns:
            y = df["Year_Quarter"].astype(str).str.extract(r"(\d{4})", expand=False)
            df["year"] = pd.to_numeric(y, errors="coerce")
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "quarter_int" not in df.columns:
        if "quarter" in df.columns:
            q = pd.to_numeric(df["quarter"], errors="coerce")
            if q.dropna().between(0, 3).all():
                q = q.replace({0:1, 1:2, 2:3, 3:4})
            df["quarter_int"] = q
        elif "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            df["quarter_int"] = dt.dt.quarter
    else:
        df["quarter_int"] = pd.to_numeric(df["quarter_int"], errors="coerce")
        if df["quarter_int"].dropna().between(0, 3).all():
            df["quarter_int"] = df["quarter_int"].replace({0:1, 1:2, 2:3, 3:4})
    return df

def _attach_time_idx(df: pd.DataFrame):
    sort_keys = ["ticker", "year", "quarter_int"]
    df = df.sort_values(sort_keys).reset_index(drop=True)
    df["time_idx"] = df.groupby("ticker").cumcount()
    return df

def _apply_fixed_splits(df: pd.DataFrame):
    (ty1, tq1) = _parse_yq_str(SPLIT_TRAIN[0])
    (ty2, tq2) = _parse_yq_str(SPLIT_TRAIN[1])
    (vy1, vq1) = _parse_yq_str(SPLIT_VAL[0])
    (vy2, vq2) = _parse_yq_str(SPLIT_VAL[1])
    (sy1, sq1) = _parse_yq_str(SPLIT_TEST[0])
    (sy2, sq2) = _parse_yq_str(SPLIT_TEST[1])

    train_lo, train_hi = _yq_scalar(ty1, tq1), _yq_scalar(ty2, tq2)
    val_lo,   val_hi   = _yq_scalar(vy1, vq1), _yq_scalar(vy2, vq2)
    test_lo,  test_hi  = _yq_scalar(sy1, sq1), _yq_scalar(sy2, sq2)

    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["quarter_int"] = pd.to_numeric(df["quarter_int"], errors="coerce")
    df = df[df["year"].notna() & df["quarter_int"].notna()]

    df["yq_key"] = _yq_key_vec(df["year"], df["quarter_int"])

    df["split"] = np.where((df["yq_key"] >= train_lo) & (df["yq_key"] <= train_hi), "train",
                   np.where((df["yq_key"] >= val_lo) & (df["yq_key"] <= val_hi), "val",
                   np.where((df["yq_key"] >= test_lo) & (df["yq_key"] <= test_hi), "test", "drop")))
    df = df[df["split"] != "drop"].copy()
    return df

def _canonical_ticker(ticker: str) -> str:
    t = str(ticker).upper().strip()
    if t == "GOOG":
        return "GOOGL"
    return t


def _load_all(data_dir: str, finbert_csv: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*_feature.csv"))
    if not files:
        raise RuntimeError(f"No *_feature.csv files found in {data_dir}")

    frames = []
    for fp in files:
        # Robust case-insensitive matching
        fname = os.path.basename(fp).upper()
        if not any(f"{ticker}" in fname for ticker in MEGA_CAP_5):
            continue 
            
        try:
            df = pd.read_csv(fp)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")
            
    if not frames:
        raise RuntimeError(f"No readable feature files found for Mega-Cap 5: {MEGA_CAP_5}")
        
    data = pd.concat(frames, ignore_index=True)

    if "ticker" not in data.columns:
        if "TICKER" in data.columns:
            data = data.rename(columns={"TICKER": "ticker"})
        else:
            raise ValueError("No 'ticker' column found.")

    data["ticker"] = data["ticker"].astype(str).map(_canonical_ticker)
    data = data[data["ticker"].isin([_canonical_ticker(t) for t in MEGA_CAP_5])].copy()
    data = _ensure_year_quarter(data)

    # Enforce calendar Year_Quarter key from year/quarter_int for merge stability.
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["quarter_int"] = pd.to_numeric(data["quarter_int"], errors="coerce")
    data["Year_Quarter"] = data["year"].astype("Int64").astype(str) + "Q" + data["quarter_int"].astype("Int64").astype(str)
    
    if "revenue" not in data.columns:
        raise ValueError("Column 'revenue' is required.")
    if "revenue_log" not in data.columns:
        data["revenue_log"] = _safe_log1p(data["revenue"])

    if "revenue_log_lag1" not in data.columns:
        # Pre-sort to ensure groupby shift is correct
        data = data.sort_values(["ticker", "year", "quarter_int"])
        data["revenue_log_lag1"] = data.groupby("ticker")["revenue_log"].shift(1)
    if "revenue_log_lag4" not in data.columns:
        data = data.sort_values(["ticker", "year", "quarter_int"])
        data["revenue_log_lag4"] = data.groupby("ticker")["revenue_log"].shift(4)

    # Merge quarterly FinBERT features (calendar aligned), then create lagged sentiment inputs.
    if not os.path.exists(finbert_csv):
        raise RuntimeError(f"FinBERT quarterly file not found: {finbert_csv}")

    sent = pd.read_csv(finbert_csv)
    required_sent = {"ticker", "Year_Quarter", "net_sentiment", "finbert_pos", "finbert_neg", "finbert_neu"}
    missing_sent = required_sent - set(sent.columns)
    if missing_sent:
        raise ValueError(f"FinBERT CSV missing required columns: {sorted(missing_sent)}")

    sent = sent[list(required_sent)].copy()
    sent["ticker"] = sent["ticker"].astype(str).map(_canonical_ticker)
    sent["Year_Quarter"] = sent["Year_Quarter"].astype(str).str.upper().str.strip()
    sent = sent.groupby(["ticker", "Year_Quarter"], as_index=False).mean(numeric_only=True)

    data = data.merge(sent, on=["ticker", "Year_Quarter"], how="left")

    for col in ["net_sentiment", "finbert_pos", "finbert_neg", "finbert_neu"]:
        sent_col = f"sent_{col}"
        data[sent_col] = data[col]
        data[sent_col] = data.groupby("ticker")[sent_col].shift(1)
        data[f"{sent_col}_ffill"] = data.groupby("ticker")[sent_col].ffill()
        data[sent_col] = data[sent_col].fillna(0.0)
        data[f"{sent_col}_ffill"] = data[f"{sent_col}_ffill"].fillna(0.0)

    # Keep raw merge columns out of model feature selection path.
    data = data.drop(columns=["net_sentiment", "finbert_pos", "finbert_neg", "finbert_neu"], errors="ignore")

    # Attach time_idx BEFORE splitting to ensure continuity
    data = _attach_time_idx(data)

    data = _apply_fixed_splits(data)
    return data

def choose_known_future_cols(df: pd.DataFrame):
    known_reals = ["year", "quarter_int"]
    aset_log, aset_lvl = "totalAssets_lag1_log", "totalAssets_lag1"
    eqty_log, eqty_lvl = "totalEquity_lag1_log", "totalEquity_lag1"

    if aset_log in df.columns:
        known_reals.append(aset_log)
    elif aset_lvl in df.columns:
        known_reals.append(aset_lvl)

    if eqty_log in df.columns:
        known_reals.append(eqty_log)
    elif eqty_lvl in df.columns:
        known_reals.append(eqty_lvl)

    # placeholder for FinBERT integration - dual-role forward fill
    sent_cols = [c for c in df.columns if c.startswith('sent_') and c.endswith('_ffill')]
    known_reals.extend(sent_cols)

    return known_reals

def build_feature_lists(df: pd.DataFrame):
    static_categoricals = [c for c in ["ticker", "gics_sectors"] if c in df.columns]
    known_reals = choose_known_future_cols(df)
    known_categoricals = []

    base_unknown = [
        "revenue_log",
        "revenue_log_lag1", "revenue_log_lag4",
        "grossProfit", "costOfRevenue", "operatingExpenses", "snaExpenses",
        "ebitda", "operatingIncome", "incomeBeforeTax", "netIncome",
        "totalAssets", "totalEquity",
        "grossProfitRatio", "operatingIncomeRatio", "netIncomeRatio",
    ]
    unknown_reals = [c for c in base_unknown if c in df.columns]
    unknown_reals += [c for c in df.columns if c.endswith("_yoy")]

    # placeholder for FinBERT integration - observed past sentiment
    base_sent = [c for c in df.columns if c.startswith('sent_') and not c.endswith('_ffill')]
    unknown_reals.extend(base_sent)

    unknown_reals = [c for c in dict.fromkeys(unknown_reals) if c not in known_reals]

    return static_categoricals, known_reals, known_categoricals, unknown_reals

def _sanitize_encoder_features(df: pd.DataFrame, encoder_cols: list) -> tuple[pd.DataFrame, list]:
    df = df.copy()
    added_flags = []

    # --- FIX START ---
    # Bug fixed: [np.inf, -inf] -> [np.inf, -np.inf] namespace error
    df[encoder_cols] = df[encoder_cols].replace([np.inf, -np.inf], np.nan)
    # --- FIX END ---

    for c in encoder_cols:
        na = df[c].isna()
        if not na.any():
            continue
        flag = f"{c}_nanflag"
        df[flag] = na.astype(np.int8)
        added_flags.append(flag)
        if c.endswith("_yoy"):
            df.loc[na, c] = 0.0
        else:
            med = df.groupby("ticker")[c].transform("median")
            df.loc[na, c] = med[na]
            na2 = df[c].isna()
            if na2.any():
                df.loc[na2, c] = 0.0
    return df, added_flags

# ------------------ REPLICATED EVALUATION LOGIC FROM BASELINE H4 ------------------
def _extract_preds_and_index(predict_out):
    preds, index_df = predict_out, None
    if isinstance(predict_out, (list, tuple)):
        preds = predict_out[0]
        for obj in predict_out[1:]:
            if isinstance(obj, pd.DataFrame):
                index_df = obj
                break
            if isinstance(obj, pd.Series):
                index_df = obj.to_frame()
                break
    return torch.as_tensor(preds), index_df

@torch.no_grad()
def evaluate_predictions(model, loader, tag: str, inv_trans=True):
    """
    Unified evaluate function, explicitly handles multi-horizon matrix format.
    Replacedevaluate_h4_predictions logic.
    """
    out = model.predict(loader, mode="prediction", return_index=True)
    preds_t, index_df = _extract_preds_and_index(out)
    preds = preds_t.detach().cpu().numpy()
    
    # preds shape (N, max_encoder_length) for forecasting loader
    horizon = preds.shape[1]

    ys = []
    for _x, y in loader:
        if isinstance(y, (tuple, list)) and len(y) >= 1:
            y = y[0]
        if isinstance(y, dict):
            y = y.get("target", y.get("y", y))
        ys.append(torch.as_tensor(y).detach().cpu().numpy())
    y_true = np.squeeze(np.concatenate(ys, axis=0))

    # Standard Pytorch Forecasting returns squeeze (N, H) or (N, Q, H). 
    # If regression and squeeze removed (N,), explicitly reshape to (N, 1)
    if y_true.ndim == 1 and horizon == 1:
        y_true = y_true[:, np.newaxis]
        preds = preds[:, np.newaxis]

    n = min(len(y_true), len(preds))
    if len(y_true) != len(preds):
        print(f"[WARN] pred/true length mismatch: pred={len(preds)} true={len(y_true)}; truncating to {n}")
    
    y_true = y_true[:n]
    preds  = preds[:n]
    if index_df is not None and len(index_df) != n:
        index_df = index_df.iloc[:n].copy()

    if inv_trans:
        y_true_lvl = np.expm1(y_true)
        y_pred_lvl = np.expm1(preds)
    else:
        y_true_lvl = y_true
        y_pred_lvl = preds

    denom = np.clip(np.abs(y_true_lvl), 1e-8, None)
    
    mape = np.mean(np.abs((y_true_lvl - y_pred_lvl) / denom)) * 100.0
    rmse = np.sqrt(np.mean((y_true_lvl - y_pred_lvl) ** 2))
    mae  = np.mean(np.abs(y_true_lvl - y_pred_lvl))
    print(f"[{tag}] Aggregate Multi-Horizon (H={horizon}) -> MAPE: {mape:.2f}% | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    return y_true_lvl, y_pred_lvl, index_df

# ------------------ NEW: POST-PROCESSING METRICS (Standardized Epoch table) ------------------
def _extract_and_save_per_epoch_metrics(trainer, out_dir) -> pd.DataFrame:
    """
    Post-processes the messy step-wise metrics.csv generated by CSVLogger
    into a clean, finalized per-epoch summary table in OUT_DIR.
    Returns the DF for visualization usage without re-reading file.
    """
    print("[INFO] Post-processing per-epoch metrics table...")

    # Lightning's CSVLogger creates a metrics.csv inside OUT_DIR/lightning_logs/version_X
    # Trainer(logger=csv_logger) knows which version it just finished
    version = trainer.logger.version
    raw_log_path = os.path.join(out_dir, "lightning_logs", f"version_{version}", "metrics.csv")
    final_file_path = os.path.join(out_dir, "per_epoch_metrics.csv")

    if not os.path.exists(raw_log_path):
        print(f"[WARN] Could not find raw metrics.csv at {raw_log_path}. Skipping per-epoch table generation.")
        return pd.DataFrame() # Return empty if raw not found

    try:
        # Load raw, messy data
        raw_metrics = pd.read_csv(raw_log_path)

        # --- Filter and Aggregate ---
        # Lightning's raw log is sparse: step-wise logs are dense, epoch-wise logs have NaNs elsewhere.
        # We want to aggregate by epoch. We'll use .groupby('epoch').max() which works well with
        # sparse data (picks the single non-NaN value logged at the end of the epoch).

        per_epoch_df = raw_metrics.groupby('epoch').max().reset_index()

        # --- Clean up columns ---
        # Drop the 'step' column as it's not meaningful in an epoch-level summary
        if 'step' in per_epoch_df.columns:
            per_epoch_df = per_epoch_df.drop(columns=['step'])

        # Identify and standardize key metric names
        # (e.g., 'train_loss_epoch' -> 'train_loss', 'lr-AdamW' -> 'learning_rate')
        name_mapping = {}

        # Handle loss: Pytorch Forecasting typically logs train_loss_epoch and val_loss
        if 'train_loss_epoch' in per_epoch_df.columns:
            name_mapping['train_loss_epoch'] = 'train_loss'
        elif 'train_loss' in per_epoch_df.columns: # Sometimes it's just 'train_loss'
             pass

        if 'val_loss' in per_epoch_df.columns:
             pass

        # Handle LR: LearningRateMonitor logs 'lr-OptimizerName'
        lr_cols = [c for c in per_epoch_df.columns if c.startswith('lr-')]
        if lr_cols:
            name_mapping[lr_cols[0]] = 'learning_rate' # Standardize the first LR found

        # Other metrics (e.g., validation MAE, P50 etc., will just be retained as-is)

        if name_mapping:
            per_epoch_df = per_epoch_df.rename(columns=name_mapping)

        # Define standard order for essential columns, followed by other logged metrics
        essential_cols = ['epoch', 'train_loss', 'val_loss', 'learning_rate']
        available_essential = [c for c in essential_cols if c in per_epoch_df.columns]
        other_cols = sorted([c for c in per_epoch_df.columns if c not in available_essential])

        per_epoch_df = per_epoch_df[available_essential + other_cols]

        # Save the clean table to the root output directory
        per_epoch_df.to_csv(final_file_path, index=False)
        print(f" - Saved per-epoch metrics summary (finalized) to {final_file_path}")
        
        return per_epoch_df # Return DF for immediate visualization

    except Exception as e:
        print(f"[WARN] Failed to post-process per-epoch metrics: {e}")
        return pd.DataFrame()

# ------------------ VISUALIZATION helper updated to accept in-memory DF ------------------
# This fixes Bug C (file locking / standardization)
def visualize_tft_results(metrics_df, tft_model, dataset, results_df, out_dir):
    """
    Generates standard machine learning visualizations using finalized standardized
    per_epoch_metrics DF (bypassing file read issues).
    """
    if not HAS_PLOTTING:
        print("[WARN] matplotlib/seaborn not installed. Skipping visualization outputs.")
        return

    print("[INFO] Generating visualizations...")
    
    # 1. Plot Loss Curves (Train vs Validation) using finalized standardized DF
    if not metrics_df.empty:
        plt.figure(figsize=(10, 6))
        
        # Read standardized columns - x-axis changed from 'step' to 'epoch'
        if 'train_loss' in metrics_df.columns and 'val_loss' in metrics_df.columns:
             sns.lineplot(data=metrics_df, x="epoch", y="train_loss", label="Training Loss (Huber)", color='royalblue', linewidth=2)
             sns.lineplot(data=metrics_df, x="epoch", y="val_loss", label="Validation Loss (Huber)", color='darkorange', linewidth=2, linestyle='--')
             
             plt.title("TFT Training and Validation Loss Trajectories", fontsize=14)
             plt.xlabel("Epoch", fontsize=12)
             plt.ylabel("Loss", fontsize=12)
             plt.legend()
             loss_fig_path = os.path.join(out_dir, "training_validation_loss.png")
             plt.savefig(loss_fig_path, dpi=150)
             print(f" - Saved loss curves to {loss_fig_path}")
             plt.close()
        else:
             print("[WARN] Standardization failed: 'train_loss' or 'val_loss' missing in metrics_df. Cannot plot loss.")

        # 2. Plot Learning Rate Schedule using finalized standardized DF
        if 'learning_rate' in metrics_df.columns:
             lr_log = metrics_df[['epoch', 'learning_rate']].dropna()
             if not lr_log.empty:
                  plt.figure(figsize=(10, 5))
                  sns.lineplot(data=lr_log, x="epoch", y="learning_rate", color='mediumseagreen', linewidth=2)
                  plt.title("TFT Learning Rate Schedule (Cosine Annealing)", fontsize=14)
                  plt.xlabel("Epoch", fontsize=12)
                  plt.ylabel("Learning Rate", fontsize=12)
                  plt.yscale('log') # Usually LR schedules are easier to see on log scale
                  lr_fig_path = os.path.join(out_dir, "learning_rate_schedule.png")
                  plt.savefig(lr_fig_path, dpi=150)
                  print(f" - Saved LR schedule to {lr_fig_path}")
                  plt.close()
        else:
             print("[WARN] Standardization failed: 'learning_rate' missing in metrics_df. Cannot plot LR.")
    else:
        print(f"[WARN] metrics_df is empty. Cannot plot loss/LR.")

    # 3. Plot Actual vs Predicted Revenue (Time Series for one example ticker)
    # Pick a ticker with significant movement, e.g., NVDA or AMZN
    ticker_to_plot = MEGA_CAP_5[-1] if len(MEGA_CAP_5) > 0 else "NVDA"
    company_subset = results_df[results_df["ticker"] == ticker_to_plot]
    
    if not company_subset.empty:
        company_subset = company_subset.sort_values("time_idx")
        plt.figure(figsize=(12, 6))
        # Plot Horizon 1 (next quarter) actual vs predicted
        sns.lineplot(data=company_subset, x="Year_Quarter", y="y_true_h1", label="Actual Revenue (H=1)", color='black', marker='o', markersize=5)
        sns.lineplot(data=company_subset, x="Year_Quarter", y="y_pred_h1", label="Predicted Revenue (H=1)", color='royalblue', linestyle='--', marker='x', markersize=5)
        
        plt.title(f"Test Set Predictions (H=1) vs Actual Revenue for {ticker_to_plot}", fontsize=14)
        plt.xlabel("Year_Quarter", fontsize=12)
        plt.ylabel("Revenue (Log-Transform Scale)", fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        ts_fig_path = os.path.join(out_dir, f"test_actual_vs_predicted_h1_{ticker_to_plot}.png")
        plt.savefig(ts_fig_path, dpi=150)
        print(f" - Saved actual vs predicted plot to {ts_fig_path}")
        plt.close()
    else:
        print(f"[WARN] Could not find test data for ticker {ticker_to_plot} to plot time series.")

    # 4. TFT Specific - Variable Selection Importance (Encoder)
    # Re-reading .npy is okay, file locking shouldn't affect these. Fixed by Bug A.
    interpret_enc_path = os.path.join(out_dir, "val_encoder_vars.npy")
    if os.path.exists(interpret_enc_path):
        importance_means = np.load(interpret_enc_path)
        encoder_variables = dataset.encoder_variables
        
        if len(importance_means) == len(encoder_variables):
            vsn_df = pd.DataFrame({
                'Variable': encoder_variables,
                'Importance': importance_means
            }).sort_values(by="Importance", ascending=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(data=vsn_df, x="Importance", y="Variable", palette="viridis")
            plt.title("TFT Overall Variable Selection Importance (Encoder)", fontsize=14)
            plt.xlabel("Mean Importance Weight", fontsize=12)
            vsn_fig_path = os.path.join(out_dir, "tft_encoder_variable_importance.png")
            plt.savefig(vsn_fig_path, dpi=150, bbox_inches='tight')
            print(f" - Saved encoder variable importance to {vsn_fig_path}")
            plt.close()
    else:
        print("[WARN] val_encoder_vars.npy not found (Unable to plot VSN importance).")

    # 5. TFT Specific - Attention Heatmap
    # Re-reading .npy is okay. Fixed by Bug A.
    interpret_att_path = os.path.join(out_dir, "val_attention.npy")
    if os.path.exists(interpret_att_path):
        attention_array = np.load(interpret_att_path)
        if attention_array.size > 0:
            # Standard structure: (N_samples, heads, MAX_PRED_LEN, MAX_ENCODER_LEN)
            # Aggregating across samples, heads, and future prediction length to get overall lookback focus
            if len(attention_array.shape) == 4:
                mean_attention = attention_array.mean(axis=(0, 1, 2))
                
                # Create a heatmap
                plt.figure(figsize=(8, 4))
                # Reshape to 2D matrix (1 x Lookback)
                attention_matrix = mean_attention.reshape(1, MAX_ENCODER_LEN)
                
                # Generate labels for quarters: lag12 to lag1
                quarter_labels = [f"t-{MAX_ENCODER_LEN - i}" for i in range(MAX_ENCODER_LEN)]
                
                sns.heatmap(attention_matrix, cmap="Blues", annot=True, fmt=".3f", 
                            cbar_kws={'label': 'Attention Weight'},
                            xticklabels=quarter_labels, yticklabels=["TFT Focus"])
                
                plt.title("TFT Masked Multi-Head Attention: Average Lookback Pattern", fontsize=14)
                plt.xlabel("Historical Quarters (Lags)", fontsize=12)
                attention_fig_path = os.path.join(out_dir, "tft_attention_heatmap.png")
                plt.savefig(attention_fig_path, dpi=150, bbox_inches='tight')
                print(f" - Saved attention heatmap to {attention_fig_path}")
                plt.close()
    else:
        print("[WARN] val_attention.npy not found (Unable to plot Attention Heatmap).")

# ------------------ MAIN ------------------
def main():
    seed_everything(SEED, workers=True)

    df = _load_all(DATA_DIR, FINBERT_CSV)
    print(f"[INFO] Loaded Mega-Cap 5 dataset. Shape: {df.shape}")
    print(f"[INFO] Included Tickers: {df['ticker'].unique().tolist()}")
    
    static_categoricals, known_reals, known_categoricals, unknown_reals = build_feature_lists(df)

    # Optimization: define incomplete rows once
    REQ_LAGS = [c for c in ["revenue_log", "revenue_log_lag1", "revenue_log_lag4"] if c in df.columns]

    def _prune_split(dfsplit: pd.DataFrame, name: str) -> pd.DataFrame:
        before = len(dfsplit)
        out = dfsplit.dropna(subset=REQ_LAGS).copy()
        dropped = before - len(out)
        if dropped > 0:
            print(f"[INFO] {name}: dropped {dropped} rows lacking {REQ_LAGS}")
        return out

    # Separately prune to sanitize without leakage
    df_train_base = _prune_split(df.loc[df["split"].eq("train")], "TRAIN")
    df_val_base   = _prune_split(df.loc[df["split"].eq("val")],   "VAL")
    df_test_base  = _prune_split(df.loc[df["split"].eq("test")],  "TEST")

    # PREPEND HISTORY to Validation/Test for contiguous lookback window
    # Validation needs last 12q from Train
    val_history = df_train_base.groupby("ticker").tail(MAX_ENCODER_LEN)
    # KeyError: 'time_idx' fixed because _load_all now attaches it.
    df_val_ext = pd.concat([val_history, df_val_base]).sort_values(["ticker", "time_idx"])

    # Test needs last 12q from combined history
    train_val_combined = pd.concat([df_train_base, df_val_base]).sort_values(["ticker", "time_idx"])
    test_history = train_val_combined.groupby("ticker").tail(MAX_ENCODER_LEN)
    df_test_ext = pd.concat([test_history, df_test_base]).sort_values(["ticker", "time_idx"])

    # Define sanitization columns globally based on all available inputs
    encoder_cols = list(dict.fromkeys(["revenue_log"] + unknown_reals))

    # Sanitize each split independently (Train/Val/Test data is separate here)
    df_train, flags_train = _sanitize_encoder_features(df_train_base, encoder_cols)
    df_val,   flags_val   = _sanitize_encoder_features(df_val_ext,    encoder_cols)
    df_test,  flags_test  = _sanitize_encoder_features(df_test_ext,   encoder_cols)

    # Identify all flag columns created across all splits and unify them to prevent TimeSeriesDataSet error
    flag_cols = sorted(set(flags_train) | set(flags_val) | set(flags_test))

    for col in flag_cols:
        if col not in df_train.columns: df_train[col] = 0
        if col not in df_val.columns: df_val[col] = 0
        if col not in df_test.columns: df_test[col] = 0

    # Ensure actual encoder columns have 0.0 imputation if they were skipped (should not happen if dropna worked)
    for col in encoder_cols:
        if col not in df_train.columns: df_train[col] = 0.0
        if col not in df_val.columns: df_val[col] = 0.0
        if col not in df_test.columns: df_test[col] = 0.0

    unknown_reals_extended = list(dict.fromkeys(unknown_reals + flag_cols))
 
    # ---- Build datasets
    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="revenue_log",
        group_ids=["ticker"],
        max_encoder_length=MAX_ENCODER_LEN,
        max_prediction_length=MAX_PRED_LEN,

        static_categoricals=static_categoricals,
        static_reals=[],

        time_varying_known_categoricals=known_categoricals,
        time_varying_known_reals=known_reals,

        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=list(dict.fromkeys(["revenue_log"] + unknown_reals_extended)),

        target_normalizer=None, # Modeling log(1+x), transform inversion in evaluation
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Val/Test use specialized extension to preserve contiguity
    validation = training.from_dataset(training, df_val)
    testing    = training.from_dataset(training, df_test)

    train_loader = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
    val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    test_loader  = testing.to_dataloader(train=False,  batch_size=BATCH_SIZE, num_workers=0)

    # ---- Trainer
    # Bug fix: Increased patience to 20 to allow > 100 epoch run
    early_stop = EarlyStopping(monitor="val_loss", patience=20, mode="min")
    ckpt = ModelCheckpoint(
        dirpath=OUT_DIR,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", save_top_k=1, mode="min"
    )
    lr_mon = LearningRateMonitor(logging_interval="epoch")
    
    # Configure logging to save to OUT_DIR so we can read metrics back for visualization
    csv_logger = CSVLogger(OUT_DIR, name="lightning_logs")

    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    precision = "bf16-mixed" if use_gpu else 32
    print(f"[INFO] Trainer backend: accelerator={accelerator}, precision={precision}")

    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=0.1,
        callbacks=[early_stop, ckpt, lr_mon],
        # trainer automatically adds version_0, version_1 folders inside this directory
        default_root_dir=OUT_DIR, 
        log_every_n_steps=50,
        logger=csv_logger  # Use explicitly defined CSV logger
    )

    loss = QuantileLoss()

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LR,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTN_HEADS,
        dropout=DROPOUT,
        loss=loss,
        optimizer="AdamW",
        weight_decay=WEIGHT_DECAY,
        reduce_on_plateau_patience=4,
    )

    print(f"Model size: {tft.size()/1e3:.1f}k parameters")
    trainer.fit(tft, train_loader, val_loader)

    best_path = ckpt.best_model_path
    print("Best checkpoint:", best_path)
    best = TemporalFusionTransformer.load_from_checkpoint(best_path)

    # --- NEW: Post-process training history (Standardization + fix file locking) ---
    finalized_metrics_df = pd.DataFrame()
    try:
        # Pass the trainer (to get version) and output directory
        # Fixed Bug C root cause by returning DF to main
        finalized_metrics_df = _extract_and_save_per_epoch_metrics(trainer, OUT_DIR)
    except Exception as e:
        print(f"[WARN] Failed to generate finalized per-epoch metrics file: {e}")

    # ---- Evaluate (Unified function returns inv-transforms lvl data)
    val_true,  val_pred,  val_idx  = evaluate_predictions(best, val_loader,  "VAL")
    test_true, test_pred, test_idx = evaluate_predictions(best, test_loader, "TEST")

    # ------------------ REPLICATED OUTPUT LOGIC FROM BASELINE H4 ------------------
    # Replicate tft_test_predictions.csv matrix format
    if test_idx is None:
        # Pytorch-lightning predict() rarely returns index, fallback
        test_idx = pd.DataFrame({
            "time_idx": testing.data[testing.index.time][:len(test_true)], 
            "ticker": testing.data[testing.index.group_ids[0]][:len(test_true)],
        })

    # Prepare matrix columns for output
    pred_df = test_idx.copy()
    
    # Store matrix of y_true and y_pred
    for h in range(MAX_PRED_LEN):
        h_idx = h + 1
        pred_df[f"y_true_h{h_idx}"] = test_true[:, h]
        pred_df[f"y_pred_h{h_idx}"] = test_pred[:, h]
        pred_df[f"APE_h{h_idx}"] = np.where(np.abs(pred_df[f"y_true_h{h_idx}"]) < 1e-8, np.nan, 
                                          np.abs(pred_df[f"y_true_h{h_idx}"] - pred_df[f"y_pred_h{h_idx}"]) / np.abs(pred_df[f"y_true_h{h_idx}"]) * 100.0)

    # Calculate overall aggregate metrics for per-row mapping
    aps_matrix = np.abs(test_true - test_pred) / np.clip(np.abs(test_true), 1e-8, None)
    pred_df["APE_%"] = np.mean(aps_matrix, axis=1) * 100.0

    # Merge per-horizon predictions back to df_test_base (without ext history)
    merge_keys = ["ticker", "time_idx"]
    matrix_cols = merge_keys + [c for c in pred_df.columns if c.startswith(('y_true_h', 'y_pred_h', 'APE_h', 'APE_%'))]
    
    df_test_out = pd.merge(
        df_test_base.reset_index(drop=True), # Use base, not extended history
        pred_df[matrix_cols],
        on=merge_keys,
        how="inner" # Inner join crucial to discard history extension samples
    )

    test_path = os.path.join(OUT_DIR, "tft_test_predictions.csv")
    df_test_out.to_csv(test_path, index=False)

    # Replicate tft_test_agg_by_ticker.csv format (agg metrics across all 4 horizons)
    def _agg_ticker(g):
        # Flattened matrix for aggregate metrics
        y_true_g = np.concatenate([g[f"y_true_h{h+1}"].values for h in range(MAX_PRED_LEN)])
        y_pred_g = np.concatenate([g[f"y_pred_h{h+1}"].values for h in range(MAX_PRED_LEN)])
        
        # trim nans if horizon extending past real key bounds (dropna failed somewhere)
        y_true_g = y_true_g[~np.isnan(y_true_g)]
        y_pred_g = y_pred_g[:len(y_true_g)]

        rmse = np.sqrt(np.mean((y_true_g - y_pred_g) ** 2))
        mae = np.mean(np.abs(y_true_g - y_pred_g))
        denom_g = np.clip(np.abs(y_true_g), 1e-8, None)
        mape = np.mean(np.abs((y_true_g - y_pred_g) / denom_g)) * 100.0
        
        # Collect per-horizon metrics too
        data = {"RMSE": rmse, "MAE": mae, "MAPE": mape, "N": len(g)}
        for h in range(MAX_PRED_LEN):
            h_idx = h + 1
            t_h = g[f"y_true_h{h_idx}"].values
            p_h = g[f"y_pred_h{h_idx}"].values
            # handle trims
            t_h = t_h[~np.isnan(t_h)]
            p_h = p_h[:len(t_h)]
            d_h = np.clip(np.abs(t_h), 1e-8, None)
            
            data[f"MAPE_h{h_idx}"] = np.mean(np.abs((t_h - p_h) / d_h)) * 100.0
            data[f"RMSE_h{h_idx}"] = np.sqrt(np.mean((t_h - p_h) ** 2))
        
        return pd.Series(data)

    by_ticker = df_test_out.groupby("ticker", as_index=False).apply(_agg_ticker).reset_index(drop=True)
    agg_path  = os.path.join(OUT_DIR, "tft_test_agg_by_ticker.csv")
    by_ticker.to_csv(agg_path, index=False)

    # Replicate tft_test_overall_stats.csv format
    overall = pd.DataFrame({
        "metric": ["RMSE", "MAE", "MAPE(%)"],
        "mean": [by_ticker["RMSE"].mean(), by_ticker["MAE"].mean(), by_ticker["MAPE"].mean()],
        "std":  [by_ticker["RMSE"].std(), by_ticker["MAE"].std(), by_ticker["MAPE"].std()],
        "N_tickers": [len(by_ticker)] * 3
    })
    
    # Add per-horizon overall stats
    for h in range(MAX_PRED_LEN):
        h_idx = h+1
        overall = pd.concat([overall, pd.DataFrame({
            "metric": [f"MAPE_h{h_idx}(%)"],
            "mean": [by_ticker[f"MAPE_h{h_idx}"].mean()],
            "std": [by_ticker[f"MAPE_h{h_idx}"].std()],
            "N_tickers": [len(by_ticker)]
        })], ignore_index=True)

    ov_path = os.path.join(OUT_DIR, "tft_test_overall_stats.csv")
    overall.to_csv(ov_path, index=False)

    print("Saved (matched to h4 baseline format):")
    print(" - per-row test predictions (matrix):", test_path)
    print(" - per-ticker metrics (standardized):", agg_path)
    print(" - overall mean/std:", ov_path)

    # ---- Save TFT Interpretation Artifacts ----
    try:
        # standard Pytorch Forecasting logic
        interpret = best.interpret_output(best.predict(val_loader, mode="raw"))
        
        np.save(os.path.join(OUT_DIR, "val_attention.npy"), interpret.get("attention", torch.tensor([])).detach().cpu().numpy())
        np.save(os.path.join(OUT_DIR, "val_encoder_vars.npy"), interpret.get("encoder_variables", torch.tensor([])).detach().cpu().numpy())
        np.save(os.path.join(OUT_DIR, "val_decoder_vars.npy"), interpret.get("decoder_variables", torch.tensor([])).detach().cpu().numpy())
        print("Interpretability arrays saved.")
    except Exception as e:
        print(f"[WARN] Failed to interpret/save TFT artifacts: {e}")

    # ------------------ CALL VISUALIZATIONS (Updated fixed logic) ------------------
    try:
        # Pass the finalized metrics DF (prevents lock/standardization issues)
        # Bypassed Bugs A, B, C root causes
        visualize_tft_results(finalized_metrics_df, best, training, df_test_out, OUT_DIR)
    except Exception as e:
        print(f"[WARN] Visualization failed: {e}")

if __name__ == "__main__":
    main()