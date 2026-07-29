import os

import matplotlib.pyplot as plt
import pandas as pd

from data.dataset import downsample_df


def save_prediction_plot(agg_df, output_png, max_plot_points=0):
    plot_df = downsample_df(agg_df, max_plot_points)

    plt.figure(figsize=(15, 6))
    plt.plot(plot_df["date"], plot_df["imf3"], label="Raw IMF3", linewidth=0.8, alpha=0.65)
    plt.plot(plot_df["date"], plot_df["bg_pred"], label="Learned Background", linewidth=1.1)
    plt.plot(
        plot_df["date"],
        plot_df["recon_pred"],
        label="Reconstruction = BG + Residual",
        linewidth=0.9,
        alpha=0.75,
    )
    plt.title("Self-supervised Background-Residual Decomposition")
    plt.xlabel("Time")
    plt.ylabel("IMF3")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=160)
    plt.close()


def save_residual_plot(agg_df, output_png, max_plot_points=0):
    plot_df = downsample_df(agg_df, max_plot_points)

    plt.figure(figsize=(15, 5))
    plt.plot(
        plot_df["date"],
        plot_df["residual_imf3_minus_bg_pred"],
        label="IMF3 - Learned Background",
        linewidth=1.0,
    )
    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.title("Residual Curve")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=160)
    plt.close()


def save_gate_plot(agg_df, output_png, max_plot_points=0):
    plot_df = downsample_df(agg_df, max_plot_points)

    plt.figure(figsize=(15, 4))
    plt.plot(plot_df["date"], plot_df["future_gate"], label="Residual Gate", linewidth=0.9)
    plt.title("Learned Residual Gate")
    plt.xlabel("Time")
    plt.ylabel("Gate")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=160)
    plt.close()


def save_loss_history(history, output_csv, output_png):
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if len(hist_df) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train loss", linewidth=1.2)
        plt.plot(hist_df["epoch"], hist_df["val_loss"], label="Val loss", linewidth=1.2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
        plt.savefig(output_png, dpi=160)
        plt.close()


def save_inference_overview_plot(agg_df, output_png, max_plot_points=0, show=False):
    plot_df = downsample_df(agg_df, max_plot_points)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(plot_df["date"], plot_df["imf3"], label="VMD-IMF3", linewidth=0.8, alpha=0.65)
    axes[0].plot(plot_df["date"], plot_df["bg_pred"], label="Predicted Background", linewidth=1.2)
    axes[0].plot(plot_df["date"], plot_df["recon_pred"], label="BG + Residual", linewidth=0.9, alpha=0.75)
    axes[0].set_title("VMD-IMF3 Background-Residual Inference")
    axes[0].set_ylabel("IMF3")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    axes[0].legend()

    axes[1].plot(
        plot_df["date"],
        plot_df["residual_imf3_minus_bg_pred"],
        label="Residual = IMF3 - Predicted Background",
        linewidth=1.0,
    )
    axes[1].axhline(0, linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Residual")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    axes[1].legend()

    axes[2].plot(plot_df["date"], plot_df["res_pred"], label="Predicted Residual Branch", linewidth=0.9)
    axes[2].plot(plot_df["date"], plot_df["future_gate"], label="Future Gate", linewidth=0.9, alpha=0.8)
    axes[2].set_ylabel("Res / Gate")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)
