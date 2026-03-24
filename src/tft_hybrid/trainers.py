import os
import pandas as pd
import numpy as np
import torch

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from .config import Paths, SEED
from .dataset import prepare_hybrid_frames, build_tsdatasets
from .evaluate import evaluate_predictions, save_prediction_exports

torch.set_float32_matmul_precision("high")


def extract_and_save_per_epoch_metrics(trainer, out_dir):
    version = trainer.logger.version
    raw_log_path = os.path.join(out_dir, "lightning_logs", f"version_{version}", "metrics.csv")
    final_file_path = os.path.join(out_dir, "per_epoch_metrics.csv")
    if not os.path.exists(raw_log_path):
        return pd.DataFrame()
    raw_metrics = pd.read_csv(raw_log_path)
    per_epoch_df = raw_metrics.groupby("epoch").max().reset_index()
    if "step" in per_epoch_df.columns:
        per_epoch_df = per_epoch_df.drop(columns=["step"])
    rename_map = {}
    if "train_loss_epoch" in per_epoch_df.columns:
        rename_map["train_loss_epoch"] = "train_loss"
    lr_cols = [c for c in per_epoch_df.columns if c.startswith("lr-")]
    if lr_cols:
        rename_map[lr_cols[0]] = "learning_rate"
    if rename_map:
        per_epoch_df = per_epoch_df.rename(columns=rename_map)
    per_epoch_df.to_csv(final_file_path, index=False)
    return per_epoch_df


def _build_trainer(out_dir: str, max_epochs: int = 150):
    early_stop = EarlyStopping(monitor="val_loss", patience=20, mode="min")
    ckpt = ModelCheckpoint(dirpath=out_dir, filename="tft-{epoch:02d}-{val_loss:.4f}", monitor="val_loss", save_top_k=1, mode="min")
    lr_mon = LearningRateMonitor(logging_interval="epoch")
    csv_logger = CSVLogger(out_dir, name="lightning_logs")
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    precision = "bf16-mixed" if use_gpu else 32
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        max_epochs=max_epochs,
        gradient_clip_val=0.1,
        callbacks=[early_stop, ckpt, lr_mon],
        default_root_dir=out_dir,
        log_every_n_steps=50,
        logger=csv_logger,
    )
    return trainer, ckpt


def _run_experiment(text_source: str, out_dir: str, hidden_size: int = 64, attention_heads: int = 4, dropout: float = 0.15, learning_rate: float = 1e-3, weight_decay: float = 1e-5, max_epochs: int = 150):
    seed_everything(SEED, workers=True)
    bundle = prepare_hybrid_frames(text_source)
    training, validation, testing = build_tsdatasets(bundle, max_prediction_length=4)
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    test_loader = testing.to_dataloader(train=False, batch_size=64, num_workers=0)
    trainer, ckpt = _build_trainer(out_dir, max_epochs=max_epochs)
    loss = QuantileLoss()
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_heads,
        dropout=dropout,
        loss=loss,
        optimizer="AdamW",
        weight_decay=weight_decay,
        reduce_on_plateau_patience=4,
    )
    trainer.fit(tft, train_loader, val_loader)
    best = TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path)
    extract_and_save_per_epoch_metrics(trainer, out_dir)
    _, _, _ = evaluate_predictions(best, val_loader, "VAL")
    test_true, test_pred, test_idx = evaluate_predictions(best, test_loader, "TEST")
    save_prediction_exports(bundle, testing, test_true, test_pred, test_idx, out_dir, max_pred_len=4)
    return best


def run_finbert_experiment():
    paths = Paths()
    return _run_experiment("finbert", paths.finbert_out_dir, hidden_size=64, attention_heads=4, dropout=0.15, learning_rate=1e-3, weight_decay=1e-5, max_epochs=150)


def run_llama3_experiment():
    paths = Paths()
    return _run_experiment("llama3", paths.llama3_out_dir, hidden_size=64, attention_heads=4, dropout=0.15, learning_rate=1e-3, weight_decay=1e-5, max_epochs=150)
