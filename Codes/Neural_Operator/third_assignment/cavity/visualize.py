import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def configure_times_font():
    windows_times = [
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
    ]
    loaded = [path for path in windows_times if path.exists()]
    if loaded:
        for path in loaded:
            font_manager.fontManager.addfont(str(path))
        return font_manager.FontProperties(fname=str(loaded[0])).get_name()
    return "STIXGeneral"


TIMES_FONT = configure_times_font()

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [TIMES_FONT, "Times New Roman", "Times", "STIXGeneral"],
        "font.sans-serif": [TIMES_FONT, "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


def _plot_history(ax, history, label):
    epochs = [r["epoch"] for r in history]
    train = [r["train_mse"] for r in history]
    ax.plot(epochs, train, label=f"{label} train MSE", linewidth=2)
    eval_epochs = [r["epoch"] for r in history if "test_mse" in r]
    test = [r["test_mse"] for r in history if "test_mse" in r]
    if test:
        ax.plot(eval_epochs, test, "--", label=f"{label} test MSE", linewidth=2)


def plot_loss_curves(deeponet_history, fno_history, out_file):
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_history(ax, deeponet_history, "DeepONet")
    _plot_history(ax, fno_history, "FNO")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training and Test Loss Curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(summary_rows, out_file):
    names = [r["model"] for r in summary_rows]
    rel = [r["relative_l2_mean"] for r in summary_rows]
    mse = [r["mse"] for r in summary_rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(x, rel, color=["#3465a4", "#cc7a29"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Relative L2")
    axes[0].set_title("Mean Relative L2")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, mse, color=["#3465a4", "#cc7a29"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Mean MSE")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _imshow(ax, field, x_2d, y_2d, title, cmap):
    im = ax.imshow(
        field,
        origin="lower",
        extent=[x_2d.min(), x_2d.max(), y_2d.min(), y_2d.max()],
        aspect="equal",
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return im


def plot_component_triplet(model_name, component, true, pred, x_2d, y_2d, sample_index, out_file):
    comp_idx = 0 if component == "u" else 1
    true_field = true[sample_index, :, :, comp_idx]
    pred_field = pred[sample_index, :, :, comp_idx]
    err = np.abs(pred_field - true_field)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, field, title, cmap in [
        (axes[0], true_field, f"{model_name} {component} true", "coolwarm"),
        (axes[1], pred_field, f"{model_name} {component} pred", "coolwarm"),
        (axes[2], err, f"{model_name} {component} abs error", "magma"),
    ]:
        im = _imshow(ax, field, x_2d, y_2d, title, cmap)
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_speed_triplet(model_name, true, pred, x_2d, y_2d, sample_index, out_file):
    true_speed = np.sqrt(np.sum(true[sample_index] ** 2, axis=-1))
    pred_speed = np.sqrt(np.sum(pred[sample_index] ** 2, axis=-1))
    err = np.abs(pred_speed - true_speed)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, field, title, cmap in [
        (axes[0], true_speed, f"{model_name} speed true", "viridis"),
        (axes[1], pred_speed, f"{model_name} speed pred", "viridis"),
        (axes[2], err, f"{model_name} speed abs error", "magma"),
    ]:
        im = _imshow(ax, field, x_2d, y_2d, title, cmap)
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_streamlines(model_name, pred, x_2d, y_2d, sample_index, out_file):
    u = pred[sample_index, :, :, 0]
    v = pred[sample_index, :, :, 1]
    speed = np.sqrt(u**2 + v**2)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.contourf(x_2d, y_2d, speed, levels=30, cmap="viridis")
    ax.streamplot(x_2d, y_2d, u, v, color="white", density=1.3, linewidth=0.7)
    ax.set_title(f"{model_name} predicted streamlines")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(data, deeponet_pred, fno_pred, deeponet_history, fno_history, summary_rows, out_dir, sample_index=0):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    target = data.target("test")
    plot_loss_curves(deeponet_history, fno_history, out_dir / "loss_curves.png")
    plot_metric_bars(summary_rows, out_dir / "metric_comparison.png")
    for name, pred in [("DeepONet", deeponet_pred), ("FNO", fno_pred)]:
        safe = name.lower()
        plot_component_triplet(name, "u", target, pred, data.x_2d, data.y_2d, sample_index, out_dir / f"{safe}_u_heatmap.png")
        plot_component_triplet(name, "v", target, pred, data.x_2d, data.y_2d, sample_index, out_dir / f"{safe}_v_heatmap.png")
        plot_speed_triplet(name, target, pred, data.x_2d, data.y_2d, sample_index, out_dir / f"{safe}_speed_heatmap.png")
        plot_streamlines(name, pred, data.x_2d, data.y_2d, sample_index, out_dir / f"{safe}_streamlines.png")
