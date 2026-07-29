import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import IMF3WindowDataset, StandardScaler, load_frame, split_indices
from models.losses import GATE_RATIO, TV_RATIO, geobr_decomposition_objective
from utils.evaluation import build_prediction_tables, evaluate
from utils.plotting import (
    save_gate_plot,
    save_loss_history,
    save_prediction_plot,
    save_residual_plot,
)
from utils.common import parse_kernels, resolve_project_path, set_seed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="CD_train_300.csv")
    parser.add_argument("--column", default="IMF_3")
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--kernels", default="3,7,15")
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--recon-penalty", choices=["charbonnier", "mse"], default="charbonnier")
    parser.add_argument("--charbonnier-eps", type=float, default=1e-6)
    parser.add_argument("--lambda-bg", type=float, default=0.40)
    parser.add_argument("--lambda-res", type=float, default=0.01)

    parser.add_argument("--out-dir", default="geobr_mamba_decomp_outputs")
    parser.add_argument("--model-out", default="geobr_mamba.pt")
    parser.add_argument("--pred-long-out", default="decomp_pred_long.csv")
    parser.add_argument("--pred-agg-out", default="decomp_pred_agg.csv")
    parser.add_argument("--plot-out", default="decomp_bg_prediction.png")
    parser.add_argument("--residual-plot-out", default="decomp_residual.png")
    parser.add_argument("--gate-plot-out", default="decomp_gate.png")
    parser.add_argument("--loss-csv-out", default="loss_history.csv")
    parser.add_argument("--loss-plot-out", default="loss_history.png")
    parser.add_argument("--max-plot-points", type=int, default=0)
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    from models.geobr_mamba import MambaSelfDecompBG

    set_seed(args.seed)

    kernels = parse_kernels(args.kernels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_path = resolve_project_path(args.csv, PROJECT_ROOT)
    out_dir = resolve_project_path(args.out_dir, PROJECT_ROOT)

    os.makedirs(out_dir, exist_ok=True)
    model_out = os.path.join(out_dir, args.model_out)
    pred_long_out = os.path.join(out_dir, args.pred_long_out)
    pred_agg_out = os.path.join(out_dir, args.pred_agg_out)
    plot_out = os.path.join(out_dir, args.plot_out)
    residual_plot_out = os.path.join(out_dir, args.residual_plot_out)
    gate_plot_out = os.path.join(out_dir, args.gate_plot_out)
    loss_csv_out = os.path.join(out_dir, args.loss_csv_out)
    loss_plot_out = os.path.join(out_dir, args.loss_plot_out)
    config_out = os.path.join(out_dir, "config.json")

    print(f"device: {device}")
    print(f"input csv: {csv_path}")
    print(f"input column: {args.column}")
    print("model: IMF3 -> MultiScaleCausalConv -> Mamba -> BG branch + Gated Residual branch")
    print("training: unified self-supervised decomposition, no background label")
    print(f"kernels: {kernels}")
    print(
        f"unified loss: L_GeoBR = D + {args.lambda_bg} * Omega_B + {args.lambda_res} * Omega_R, "
        f"tv_ratio={TV_RATIO}, gate_ratio={GATE_RATIO}, recon_penalty={args.recon_penalty}"
    )

    df = load_frame(csv_path, args.column)
    dates = df["date"].astype(str).to_numpy()
    values_raw = df["imf3"].to_numpy(dtype=np.float32)
    n = len(df)

    if n < args.seq_len + args.pred_len + 10:
        raise ValueError(
            f"Data is too short: len={n}, at least seq_len + pred_len + 10 rows are required."
        )

    train_end, val_end = split_indices(n, args.train_ratio, args.val_ratio)
    print(f"data rows: {n}")
    print(f"train_end={train_end}, val_end={val_end}, test_start={val_end}")

    scaler = StandardScaler()
    scaler.fit(values_raw[:train_end])
    values_norm = scaler.transform(values_raw).astype(np.float32)
    print(f"scaler mean={scaler.mean:.8f}, std={scaler.std:.8f}")

    train_ds = IMF3WindowDataset(
        values_norm=values_norm,
        values_raw=values_raw,
        dates=dates,
        start=0,
        end=train_end,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    val_ds = IMF3WindowDataset(
        values_norm=values_norm,
        values_raw=values_raw,
        dates=dates,
        start=max(0, train_end - args.seq_len),
        end=val_end,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    test_ds = IMF3WindowDataset(
        values_norm=values_norm,
        values_raw=values_raw,
        dates=dates,
        start=max(0, val_end - args.seq_len),
        end=n,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )

    print(f"samples train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            "One of the datasets has zero samples. Reduce --seq-len/--pred-len or adjust split ratios."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MambaSelfDecompBG(
        d_model=args.d_model,
        pred_len=args.pred_len,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        kernels=kernels,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "kernels": kernels,
                "scaler": scaler.to_dict(),
                "device": device,
                "loss_type": "L_GeoBR = D + lambda_bg * Omega_B + lambda_res * Omega_R",
                "tv_ratio": TV_RATIO,
                "gate_ratio": GATE_RATIO,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        item_sum = None

        batches = tqdm(train_loader, desc=f"epoch {epoch:03d}/{args.epochs}", leave=False) if tqdm else train_loader

        for xb, yb, _, _, _ in batches:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            out = model(xb)
            loss, items = geobr_decomposition_objective(out, xb, yb, args)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            bs = xb.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

            if item_sum is None:
                item_sum = {k: 0.0 for k in items}
            for k, v in items.items():
                item_sum[k] += v * bs

            if tqdm is not None:
                batches.set_postfix(loss=f"{loss.item():.6f}")

        train_loss = train_loss_sum / max(train_count, 1)
        train_items = {k: v / max(train_count, 1) for k, v in item_sum.items()}

        val_loss, val_items, *_ = evaluate(model, val_loader, device, args, scaler)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_items.items()},
            **{f"val_{k}": v for k, v in val_items.items()},
        }
        history.append(row)

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"D={train_items['data_consistency']:.5f} "
            f"Omega_B={train_items['background_prior']:.5f} "
            f"Omega_R={train_items['residual_prior']:.5f} "
            f"obs_seq={train_items['obs_seq']:.5f} "
            f"obs_future={train_items['obs_future']:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    save_loss_history(history, loss_csv_out, loss_plot_out)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_items, pred_bg_norm, pred_res_norm, pred_recon_norm, raw_future_norm, raw_future, gates, starts = evaluate(
        model, test_loader, device, args, scaler
    )

    long_df, agg_df = build_prediction_tables(
        pred_bg_norm=pred_bg_norm,
        pred_res_norm=pred_res_norm,
        pred_recon_norm=pred_recon_norm,
        raw_future=raw_future,
        gates=gates,
        starts=starts,
        dates=dates,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        scaler=scaler,
        aggregate=args.aggregate,
    )

    long_df.to_csv(pred_long_out, index=False, encoding="utf-8-sig")
    agg_df.to_csv(pred_agg_out, index=False, encoding="utf-8-sig")

    print()
    print("========== Test Summary ==========")
    print(f"test_loss      : {test_loss:.8f}")
    for k, v in test_items.items():
        print(f"{k:15s}: {v:.8f}")
    print(f"long rows      : {len(long_df)}")
    print(f"agg rows       : {len(agg_df)}")
    print("==================================")
    print()

    save_prediction_plot(agg_df, plot_out, max_plot_points=args.max_plot_points)
    save_residual_plot(agg_df, residual_plot_out, max_plot_points=args.max_plot_points)
    save_gate_plot(agg_df, gate_plot_out, max_plot_points=args.max_plot_points)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "d_model": args.d_model,
            "d_state": args.d_state,
            "d_conv": args.d_conv,
            "expand": args.expand,
            "kernels": kernels,
            "dropout": args.dropout,
            "column": args.column,
            "mean": scaler.mean,
            "std": scaler.std,
            "model_type": "mamba_unified_self_supervised_background_residual_decomposition",
            "loss_type": "L_GeoBR = D + lambda_bg * Omega_B + lambda_res * Omega_R",
            "tv_ratio": TV_RATIO,
            "gate_ratio": GATE_RATIO,
            "args": vars(args),
            "best_val": best_val,
        },
        model_out,
    )

    print(f"saved model          : {model_out}")
    print(f"saved long pred csv  : {pred_long_out}")
    print(f"saved agg pred csv   : {pred_agg_out}")
    print(f"saved pred figure    : {plot_out}")
    print(f"saved residual figure: {residual_plot_out}")
    print(f"saved gate figure    : {gate_plot_out}")
    print(f"saved loss csv       : {loss_csv_out}")
    print(f"saved loss figure    : {loss_plot_out}")
    print(f"saved config         : {config_out}")


if __name__ == "__main__":
    main()
