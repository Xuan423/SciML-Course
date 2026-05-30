from pathlib import Path
import os

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_loss_curves(history, out_file):
    epochs = [h["epoch"] for h in history]
    total = [h["total"] for h in history]
    loss_f = [h["loss_f"] for h in history]
    loss_ic = [h["loss_ic"] for h in history]
    loss_bc = [h["loss_bc"] for h in history]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))
    ax.plot(epochs, total, label="total", linewidth=2)
    ax.plot(epochs, loss_f, label="pde", linewidth=1.5)
    ax.plot(epochs, loss_ic, label="ic", linewidth=1.5)
    ax.plot(epochs, loss_bc, label="bc", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_solution_heatmap(x, t, u_pred, out_file, u_ref=None):
    x = np.asarray(x).reshape(-1)
    t = np.asarray(t).reshape(-1)

    ncols = 3 if u_ref is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2))
    if ncols == 1:
        axes = [axes]

    im0 = axes[0].imshow(
        u_pred,
        extent=[x.min(), x.max(), t.min(), t.max()],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
    )
    axes[0].set_title("PINN prediction")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0], shrink=0.9)

    if u_ref is not None:
        im1 = axes[1].imshow(
            u_ref,
            extent=[x.min(), x.max(), t.min(), t.max()],
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
        )
        axes[1].set_title("Reference")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("t")
        fig.colorbar(im1, ax=axes[1], shrink=0.9)

        abs_err = np.abs(u_pred - u_ref)
        im2 = axes[2].imshow(
            abs_err,
            extent=[x.min(), x.max(), t.min(), t.max()],
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        axes[2].set_title("|Error|")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("t")
        fig.colorbar(im2, ax=axes[2], shrink=0.9)

    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_slices(x, t, u_pred, slice_times, out_file, u_ref=None):
    x = np.asarray(x).reshape(-1)
    t = np.asarray(t).reshape(-1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))

    for ts in slice_times:
        idx = int(np.argmin(np.abs(t - ts)))
        ax.plot(x, u_pred[idx, :], linewidth=2, label=f"pred t={t[idx]:.2f}")
        if u_ref is not None:
            ax.plot(x, u_ref[idx, :], "--", linewidth=1.5, label=f"ref t={t[idx]:.2f}")

    ax.set_title("Solution slices")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_adaptive_points(before_pts, after_pts, out_file):
    x0, t0 = before_pts
    x1, t1 = after_pts
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].scatter(x0, t0, s=5, alpha=0.6, c="#1f77b4")
    axes[0].set_title(f"Before adaptive points (N={len(x0)})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_xlim([-1.05, 1.05])
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].grid(alpha=0.3)

    axes[1].scatter(x1, t1, s=5, alpha=0.6, c="#d62728")
    axes[1].set_title(f"After adaptive points (N={len(x1)})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_xlim([-1.05, 1.05])
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
