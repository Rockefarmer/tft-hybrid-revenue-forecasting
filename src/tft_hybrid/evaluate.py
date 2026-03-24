import os
import numpy as np
import pandas as pd
import torch


def extract_preds_and_index(predict_out):
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
def evaluate_predictions(model, loader, tag: str, inv_trans: bool = True):
    out = model.predict(loader, mode="prediction", return_index=True)
    preds_t, index_df = extract_preds_and_index(out)
    preds = preds_t.detach().cpu().numpy()
    horizon = preds.shape[1]
    ys = []
    for _x, y in loader:
        if isinstance(y, (tuple, list)) and len(y) >= 1:
            y = y[0]
        if isinstance(y, dict):
            y = y.get("target", y.get("y", y))
        ys.append(torch.as_tensor(y).detach().cpu().numpy())
    y_true = np.squeeze(np.concatenate(ys, axis=0))
    if y_true.ndim == 1 and horizon == 1:
        y_true = y_true[:, np.newaxis]
        preds = preds[:, np.newaxis]
    n = min(len(y_true), len(preds))
    y_true = y_true[:n]
    preds = preds[:n]
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
    mae = np.mean(np.abs(y_true_lvl - y_pred_lvl))
    print(f"[{tag}] Aggregate Multi-Horizon (H={horizon}) -> MAPE: {mape:.2f}% | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return y_true_lvl, y_pred_lvl, index_df


def save_prediction_exports(bundle: dict, testing, test_true, test_pred, test_idx, out_dir: str, max_pred_len: int = 4):
    os.makedirs(out_dir, exist_ok=True)
    if test_idx is None:
        raise RuntimeError("Prediction index is missing; expected return_index=True output.")
    pred_df = test_idx.copy()
    for h in range(max_pred_len):
        idx = h + 1
        pred_df[f"y_true_h{idx}"] = test_true[:, h]
        pred_df[f"y_pred_h{idx}"] = test_pred[:, h]
        pred_df[f"APE_h{idx}"] = np.where(
            np.abs(pred_df[f"y_true_h{idx}"]) < 1e-8,
            np.nan,
            np.abs(pred_df[f"y_true_h{idx}"] - pred_df[f"y_pred_h{idx}"]) / np.abs(pred_df[f"y_true_h{idx}"]) * 100.0,
        )
    aps_matrix = np.abs(test_true - test_pred) / np.clip(np.abs(test_true), 1e-8, None)
    pred_df["APE_%"] = np.mean(aps_matrix, axis=1) * 100.0
    merge_keys = ["ticker", "time_idx"]
    matrix_cols = merge_keys + [c for c in pred_df.columns if c.startswith(("y_true_h", "y_pred_h", "APE_h", "APE_%"))]
    df_test_out = pd.merge(bundle["test_base"].reset_index(drop=True), pred_df[matrix_cols], on=merge_keys, how="inner")
    test_path = os.path.join(out_dir, "tft_test_predictions.csv")
    df_test_out.to_csv(test_path, index=False)

    def _agg_ticker(g):
        y_true_g = np.concatenate([g[f"y_true_h{h+1}"].values for h in range(max_pred_len)])
        y_pred_g = np.concatenate([g[f"y_pred_h{h+1}"].values for h in range(max_pred_len)])
        y_true_g = y_true_g[~np.isnan(y_true_g)]
        y_pred_g = y_pred_g[:len(y_true_g)]
        rmse = np.sqrt(np.mean((y_true_g - y_pred_g) ** 2))
        mae = np.mean(np.abs(y_true_g - y_pred_g))
        denom_g = np.clip(np.abs(y_true_g), 1e-8, None)
        mape = np.mean(np.abs((y_true_g - y_pred_g) / denom_g)) * 100.0
        data = {"RMSE": rmse, "MAE": mae, "MAPE": mape, "N": len(g)}
        for h in range(max_pred_len):
            idx = h + 1
            t_h = g[f"y_true_h{idx}"].values
            p_h = g[f"y_pred_h{idx}"].values
            t_h = t_h[~np.isnan(t_h)]
            p_h = p_h[:len(t_h)]
            d_h = np.clip(np.abs(t_h), 1e-8, None)
            data[f"MAPE_h{idx}"] = np.mean(np.abs((t_h - p_h) / d_h)) * 100.0
            data[f"RMSE_h{idx}"] = np.sqrt(np.mean((t_h - p_h) ** 2))
        return pd.Series(data)

    by_ticker = df_test_out.groupby("ticker", as_index=False).apply(_agg_ticker).reset_index(drop=True)
    by_ticker.to_csv(os.path.join(out_dir, "tft_test_agg_by_ticker.csv"), index=False)
    overall = pd.DataFrame({
        "metric": ["RMSE", "MAE", "MAPE(%)"],
        "mean": [by_ticker["RMSE"].mean(), by_ticker["MAE"].mean(), by_ticker["MAPE"].mean()],
        "std": [by_ticker["RMSE"].std(), by_ticker["MAE"].std(), by_ticker["MAPE"].std()],
        "N_tickers": [len(by_ticker)] * 3,
    })
    for h in range(max_pred_len):
        idx = h + 1
        overall = pd.concat([
            overall,
            pd.DataFrame({
                "metric": [f"MAPE_h{idx}(%)"],
                "mean": [by_ticker[f"MAPE_h{idx}"].mean()],
                "std": [by_ticker[f"MAPE_h{idx}"].std()],
                "N_tickers": [len(by_ticker)],
            }),
        ], ignore_index=True)
    overall.to_csv(os.path.join(out_dir, "tft_test_overall_stats.csv"), index=False)
    return df_test_out
