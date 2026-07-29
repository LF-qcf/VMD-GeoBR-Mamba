import numpy as np
import pandas as pd
import torch

from models.losses import geobr_decomposition_objective


@torch.no_grad()
def evaluate(model, loader, device, args, scaler):
    model.eval()

    total_loss = 0.0
    total_count = 0
    item_sum = None

    starts = []
    pred_bg_norm = []
    pred_res_norm = []
    pred_recon_norm = []
    raw_future_norm = []
    raw_future = []
    gates = []

    for xb, yb, _, y_raw, sb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        out = model(xb)
        loss, items = geobr_decomposition_objective(out, xb, yb, args)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_count += bs

        if item_sum is None:
            item_sum = {k: 0.0 for k in items}
        for k, v in items.items():
            item_sum[k] += v * bs

        bg_f = out["bg_future"].detach().cpu().numpy()
        res_f = out["res_future"].detach().cpu().numpy()
        recon_f = bg_f + res_f

        pred_bg_norm.append(bg_f)
        pred_res_norm.append(res_f)
        pred_recon_norm.append(recon_f)
        raw_future_norm.append(yb.detach().cpu().numpy())
        raw_future.append(y_raw.numpy())
        gates.append(out["future_gate"].detach().cpu().numpy())
        starts.extend(sb.detach().cpu().numpy().tolist())

    if total_count == 0:
        raise RuntimeError(
            "Evaluation loader is empty. Check data length, seq_len, pred_len, and split ratios."
        )

    pred_bg_norm = np.concatenate(pred_bg_norm, axis=0)
    pred_res_norm = np.concatenate(pred_res_norm, axis=0)
    pred_recon_norm = np.concatenate(pred_recon_norm, axis=0)
    raw_future_norm = np.concatenate(raw_future_norm, axis=0)
    raw_future = np.concatenate(raw_future, axis=0)
    gates = np.concatenate(gates, axis=0)
    starts = np.asarray(starts)

    avg_items = {k: v / max(total_count, 1) for k, v in item_sum.items()}

    return (
        total_loss / max(total_count, 1),
        avg_items,
        pred_bg_norm,
        pred_res_norm,
        pred_recon_norm,
        raw_future_norm,
        raw_future,
        gates,
        starts,
    )


def build_prediction_tables(
    pred_bg_norm,
    pred_res_norm,
    pred_recon_norm,
    raw_future,
    gates,
    starts,
    dates,
    seq_len,
    pred_len,
    scaler,
    aggregate="mean",
):
    bg_pred = scaler.inverse_transform(pred_bg_norm)
    res_pred = pred_res_norm * scaler.std
    recon_pred = scaler.inverse_transform(pred_recon_norm)

    rows = []
    for i in range(len(starts)):
        s = int(starts[i])
        for step in range(pred_len):
            target_idx = s + seq_len + step
            if target_idx >= len(dates):
                continue

            imf3_value = float(raw_future[i, step])
            bg_value = float(bg_pred[i, step])
            residual_magnitude = float(res_pred[i, step])
            recon_value = float(recon_pred[i, step])
            gate_value = float(gates[i, step])

            rows.append(
                {
                    "sample_index": i,
                    "input_start_index": s,
                    "input_end_index": s + seq_len - 1,
                    "target_index": target_idx,
                    "date": dates[target_idx],
                    "forecast_step": step + 1,
                    "imf3": imf3_value,
                    "bg_pred": bg_value,
                    "res_pred": residual_magnitude,
                    "recon_pred": recon_value,
                    "future_gate": gate_value,
                    "residual_imf3_minus_bg_pred": imf3_value - bg_value,
                    "abs_residual": abs(imf3_value - bg_value),
                    "recon_error": imf3_value - recon_value,
                }
            )

    long_df = pd.DataFrame(rows)
    if len(long_df) == 0:
        return long_df, long_df

    if aggregate == "median":
        agg_bg = long_df.groupby("target_index")["bg_pred"].median()
        agg_res = long_df.groupby("target_index")["res_pred"].median()
        agg_recon = long_df.groupby("target_index")["recon_pred"].median()
    else:
        agg_bg = long_df.groupby("target_index")["bg_pred"].mean()
        agg_res = long_df.groupby("target_index")["res_pred"].mean()
        agg_recon = long_df.groupby("target_index")["recon_pred"].mean()

    agg_df = (
        long_df.groupby("target_index")
        .agg(
            date=("date", "first"),
            imf3=("imf3", "first"),
            future_gate=("future_gate", "mean"),
            num_predictions=("bg_pred", "count"),
            min_step=("forecast_step", "min"),
            max_step=("forecast_step", "max"),
        )
        .reset_index()
    )

    agg_df["bg_pred"] = agg_df["target_index"].map(agg_bg)
    agg_df["res_pred"] = agg_df["target_index"].map(agg_res)
    agg_df["recon_pred"] = agg_df["target_index"].map(agg_recon)
    agg_df["residual_imf3_minus_bg_pred"] = agg_df["imf3"] - agg_df["bg_pred"]
    agg_df["abs_residual"] = np.abs(agg_df["residual_imf3_minus_bg_pred"])
    agg_df["recon_error"] = agg_df["imf3"] - agg_df["recon_pred"]
    agg_df["date"] = pd.to_datetime(agg_df["date"])

    return long_df, agg_df
