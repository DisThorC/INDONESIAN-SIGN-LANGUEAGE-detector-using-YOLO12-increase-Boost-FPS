from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_results_csv(results_csv: str | Path, out_path: str | Path | None = None) -> Path:
    """Plot key training curves from Ultralytics results.csv.

    Returns the saved figure path.
    """
    results_csv = Path(results_csv)
    df = pd.read_csv(results_csv)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    if "metrics/mAP50-95(B)" in df.columns:
        axes[0].plot(df.index, df["metrics/mAP50-95(B)"])
        axes[0].set_title("mAP50-95 (B)")

    if "metrics/precision(B)" in df.columns:
        axes[1].plot(df.index, df["metrics/precision(B)"])
        axes[1].set_title("Precision (B)")

    if "metrics/recall(B)" in df.columns:
        axes[2].plot(df.index, df["metrics/recall(B)"])
        axes[2].set_title("Recall (B)")

    if "train/box_loss" in df.columns:
        axes[3].plot(df.index, df["train/box_loss"], label="box")
    if "train/cls_loss" in df.columns:
        axes[3].plot(df.index, df["train/cls_loss"], label="cls")
    axes[3].set_title("Train losses")
    axes[3].legend()

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)

    out_path = Path(out_path) if out_path else results_csv.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
