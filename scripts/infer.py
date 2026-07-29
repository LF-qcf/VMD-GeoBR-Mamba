import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import StandardScaler, load_frame
from utils.common import parse_kernels, resolve_project_path
from utils.evaluation import build_prediction_tables
from utils.inference import IMF3InferDataset, get_ckpt_value, run_inference


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="data/raw/LD_CD_test_300.csv")
    parser.add_argument(
        "--column",
        default=None,
        help="Default: read from checkpoint; fallback is IMF_3.",
    )
    parser.add_argument("--ckpt", default="geobr_mamba_decomp_outputs/geobr_mamba.pt")
    parser.add_argument("--out-dir", default="geobr_mamba_infer_outputs")

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)

    parser.add_argument("--long-out", default="decomp_pred_long.csv")
    parser.add_argument("--agg-out", default="decomp_pred_agg.csv")
    parser.add_argument("--plot-out", default="infer_bg_residual_recon.png")
    parser.add_argument("--max-plot-points", type=int, default=0)

    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.show:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")

    from models.geobr_mamba import MambaSelfDecompBG
    from utils.plotting import save_inference_overview_plot

    csv_path = resolve_project_path(args.csv, PROJECT_ROOT)
    ckpt_path = resolve_project_path(args.ckpt, PROJECT_ROOT)
    out_dir = resolve_project_path(args.out_dir, PROJECT_ROOT)

    os.makedirs(out_dir, exist_ok=True)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA was requested but is not available; using CPU.")
        device = "cpu"
    else:
        device = args.device

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    seq_len = int(get_ckpt_value(ckpt, "seq_len", 300))
    pred_len = int(get_ckpt_value(ckpt, "pred_len", 1))
    d_model = int(get_ckpt_value(ckpt, "d_model", 32))
    d_state = int(get_ckpt_value(ckpt, "d_state", 16))
    d_conv = int(get_ckpt_value(ckpt, "d_conv", 4))
    expand = int(get_ckpt_value(ckpt, "expand", 2))
    kernels = parse_kernels(get_ckpt_value(ckpt, "kernels", (3, 7, 15)))
    dropout = float(get_ckpt_value(ckpt, "dropout", 0.0))
    column = args.column or get_ckpt_value(ckpt, "column", "IMF_3")

    mean = get_ckpt_value(ckpt, "mean", None)
    std = get_ckpt_value(ckpt, "std", None)
    if mean is None or std is None:
        raise RuntimeError("Checkpoint does not contain mean/std for inverse normalization.")

    print("========== Inference Config ==========")
    print(f"device   : {device}")
    print(f"csv      : {csv_path}")
    print(f"column   : {column}")
    print(f"ckpt     : {ckpt_path}")
    print(f"seq_len  : {seq_len}")
    print(f"pred_len : {pred_len}")
    print(f"d_model  : {d_model}")
    print(f"kernels  : {kernels}")
    print(f"mean/std : {float(mean):.8f} / {float(std):.8f}")
    print("======================================")

    df = load_frame(csv_path, column)
    dates = df["date"].astype(str).to_numpy()
    values_raw = df["imf3"].to_numpy(dtype=np.float32)

    scaler = StandardScaler(mean=mean, std=std)
    values_norm = scaler.transform(values_raw).astype(np.float32)

    dataset = IMF3InferDataset(
        values_norm=values_norm,
        values_raw=values_raw,
        dates=dates,
        seq_len=seq_len,
        pred_len=pred_len,
        start=args.start_index,
        end=args.end_index if args.end_index is not None else len(values_raw),
    )

    if len(dataset) == 0:
        raise RuntimeError("Inference dataset has zero samples. Check data length, seq_len, pred_len, and start/end.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MambaSelfDecompBG(
        d_model=d_model,
        pred_len=pred_len,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        kernels=kernels,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)

    (
        pred_bg_norm,
        pred_res_norm,
        pred_recon_norm,
        raw_future_norm,
        raw_future,
        gates,
        starts,
    ) = run_inference(model, loader, device)

    long_df, agg_df = build_prediction_tables(
        pred_bg_norm=pred_bg_norm,
        pred_res_norm=pred_res_norm,
        pred_recon_norm=pred_recon_norm,
        raw_future=raw_future,
        gates=gates,
        starts=starts,
        dates=dates,
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
        aggregate=args.aggregate,
    )

    long_out = os.path.join(out_dir, args.long_out)
    agg_out = os.path.join(out_dir, args.agg_out)
    plot_out = os.path.join(out_dir, args.plot_out)

    long_df.to_csv(long_out, index=False, encoding="utf-8-sig")
    agg_df.to_csv(agg_out, index=False, encoding="utf-8-sig")

    print()
    print("========== Inference Summary ==========")
    print(f"samples        : {len(dataset)}")
    print(f"long rows      : {len(long_df)}")
    print(f"agg rows       : {len(agg_df)}")
    print(f"saved long csv : {long_out}")
    print(f"saved agg csv  : {agg_out}")
    print("=======================================")

    save_inference_overview_plot(
        agg_df=agg_df,
        output_png=plot_out,
        max_plot_points=args.max_plot_points,
        show=args.show,
    )
    print(f"saved figure   : {plot_out}")


if __name__ == "__main__":
    main()
