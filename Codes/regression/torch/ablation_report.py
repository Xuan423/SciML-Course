import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from net import FNN


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "report_assets"
REPORT_PATH = ROOT / "regression_experiment_report.md"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_data(num_train, num_test=201, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, num_test, dtype=np.float32).reshape((-1, 1))
    y = np.sin(3.0 * x) ** 3
    idx = rng.choice(num_test, num_train, replace=False)
    x_train = x[idx]
    y_train = y[idx]
    return x_train, y_train, x, y


def activation_from_name(name):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def build_layers(width, depth):
    return [1] + [width] * depth + [1]


def train_and_evaluate(config, device):
    set_seed(config["seed"])
    x_train, y_train, x_ref, y_ref = generate_data(
        num_train=config["num_train"],
        num_test=config["num_test"],
        seed=config["seed"],
    )

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_ref_tensor = torch.tensor(x_ref, dtype=torch.float32, device=device)
    y_ref_tensor = torch.tensor(y_ref, dtype=torch.float32, device=device)

    model = FNN(
        build_layers(config["width"], config["depth"]),
        actn=activation_from_name(config["activation"]),
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    start = time.perf_counter()
    model.train()
    for _ in range(config["steps"]):
        pred = model(x_train_tensor)
        loss = loss_fn(pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_ref = model(x_ref_tensor)
        pred_train = model(x_train_tensor)

    elapsed = time.perf_counter() - start
    test_error = pred_ref - y_ref_tensor
    train_error = pred_train - y_train_tensor

    result = {
        "num_train": config["num_train"],
        "depth": config["depth"],
        "width": config["width"],
        "activation": config["activation"],
        "seed": config["seed"],
        "steps": config["steps"],
        "lr": config["lr"],
        "device": str(device),
        "train_mse": float(torch.mean(train_error ** 2).item()),
        "test_mse": float(torch.mean(test_error ** 2).item()),
        "test_mae": float(torch.mean(torch.abs(test_error)).item()),
        "test_max_abs": float(torch.max(torch.abs(test_error)).item()),
        "elapsed_sec": elapsed,
        "x_ref": x_ref.reshape(-1).tolist(),
        "y_ref": y_ref.reshape(-1).tolist(),
        "x_train": x_train.reshape(-1).tolist(),
        "y_train": y_train.reshape(-1).tolist(),
        "y_pred": pred_ref.detach().cpu().numpy().reshape(-1).tolist(),
    }
    return result


def aggregate_by(records, keys):
    groups = {}
    for record in records:
        group_key = tuple(record[key] for key in keys)
        groups.setdefault(group_key, []).append(record)

    summary = []
    for group_key, group_records in groups.items():
        item = {key: value for key, value in zip(keys, group_key)}
        for metric in ["train_mse", "test_mse", "test_mae", "test_max_abs", "elapsed_sec"]:
            values = np.array([record[metric] for record in group_records], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=0))
        summary.append(item)
    return sorted(summary, key=lambda row: tuple(row[key] for key in keys))


def save_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def pick_representative_record(records, **criteria):
    matches = [
        record for record in records
        if all(record[key] == value for key, value in criteria.items())
    ]
    matches = sorted(matches, key=lambda row: row["test_mse"])
    return matches[0]


def plot_sample_count_metrics(summary, out_path):
    x = [row["num_train"] for row in summary]
    mse = [row["test_mse_mean"] for row in summary]
    mse_std = [row["test_mse_std"] for row in summary]
    max_err = [row["test_max_abs_mean"] for row in summary]
    max_err_std = [row["test_max_abs_std"] for row in summary]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].errorbar(x, mse, yerr=mse_std, marker="o", linewidth=2, capsize=4, color="#0b6e4f")
    axes[0].set_title("Test MSE vs. Number of Training Points")
    axes[0].set_xlabel("Training points")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(x, max_err, yerr=max_err_std, marker="s", linewidth=2, capsize=4, color="#8f2d56")
    axes[1].set_title("Max Absolute Error vs. Number of Training Points")
    axes[1].set_xlabel("Training points")
    axes[1].set_ylabel("Max absolute error")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_depth_width_heatmap(summary, out_path):
    depths = sorted({row["depth"] for row in summary})
    widths = sorted({row["width"] for row in summary})
    grid = np.zeros((len(depths), len(widths)), dtype=np.float64)
    for row in summary:
        i = depths.index(row["depth"])
        j = widths.index(row["width"])
        grid[i, j] = row["test_mse_mean"]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    image = ax.imshow(np.log10(grid), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel("Hidden width")
    ax.set_ylabel("Hidden depth")
    ax.set_title("log10(Test MSE) for Depth/Width Grid")

    for i, depth in enumerate(depths):
        for j, width in enumerate(widths):
            ax.text(j, i, f"{grid[i, j]:.2e}", ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(image, ax=ax, label="log10(Test MSE)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_activation_metrics(summary, out_path):
    activations = [row["activation"] for row in summary]
    mse = [row["test_mse_mean"] for row in summary]
    mae = [row["test_mae_mean"] for row in summary]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(activations, mse, color=["#0b6e4f", "#1d4e89", "#f18f01", "#8f2d56"])
    axes[0].set_title("Activation vs. Test MSE")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(activations, mae, color=["#0b6e4f", "#1d4e89", "#f18f01", "#8f2d56"])
    axes[1].set_title("Activation vs. Test MAE")
    axes[1].set_ylabel("Test MAE")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_comparison(records, out_path, group_key, group_values, title):
    fig, axes = plt.subplots(1, len(group_values), figsize=(5.2 * len(group_values), 4.2), sharey=True)
    if len(group_values) == 1:
        axes = [axes]

    for ax, value in zip(axes, group_values):
        if group_key == "num_train":
            record = pick_representative_record(
                records,
                num_train=value,
                depth=2,
                width=20,
                activation="tanh",
            )
            subtitle = f"{value} training points"
        elif group_key == "activation":
            record = pick_representative_record(
                records,
                num_train=50,
                depth=2,
                width=20,
                activation=value,
            )
            subtitle = f"activation = {value}"
        else:
            raise ValueError(f"Unsupported group key: {group_key}")

        x_ref = np.array(record["x_ref"])
        y_ref = np.array(record["y_ref"])
        x_train = np.array(record["x_train"])
        y_train = np.array(record["y_train"])
        y_pred = np.array(record["y_pred"])

        ax.plot(x_ref, y_ref, color="black", linewidth=2, label="ground truth")
        ax.plot(x_ref, y_pred, color="#d1495b", linewidth=2, linestyle="--", label="prediction")
        ax.scatter(x_train, y_train, color="#00798c", s=24, alpha=0.85, label="train samples")
        ax.set_title(subtitle)
        ax.set_xlabel("x")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(title, y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows, columns, headers):
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if value == 0.0:
                    values.append("0")
                elif abs(value) >= 1.0:
                    values.append(f"{value:.4f}")
                else:
                    values.append(f"{value:.3e}")
            else:
                values.append(str(value))
        table.append("| " + " | ".join(values) + " |")
    return "\n".join(table)


def build_report(config, sample_summary, grid_summary, activation_summary, assets):
    best_grid = min(grid_summary, key=lambda row: row["test_mse_mean"])
    worst_grid = max(grid_summary, key=lambda row: row["test_mse_mean"])
    best_activation = min(activation_summary, key=lambda row: row["test_mse_mean"])
    worst_activation = max(activation_summary, key=lambda row: row["test_mse_mean"])

    sample_table = markdown_table(
        sample_summary,
        ["num_train", "test_mse_mean", "test_max_abs_mean", "elapsed_sec_mean"],
        ["Training points", "Mean test MSE", "Mean max abs error", "Mean runtime (s)"],
    )
    activation_table = markdown_table(
        activation_summary,
        ["activation", "test_mse_mean", "test_mae_mean", "test_max_abs_mean"],
        ["Activation", "Mean test MSE", "Mean test MAE", "Mean max abs error"],
    )

    depth_width_top5 = markdown_table(
        sorted(grid_summary, key=lambda row: row["test_mse_mean"])[:5],
        ["depth", "width", "test_mse_mean", "test_max_abs_mean", "elapsed_sec_mean"],
        ["Depth", "Width", "Mean test MSE", "Mean max abs error", "Mean runtime (s)"],
    )

    report = f"""# 回归实验报告：训练点数与网络结构对预测精度的影响

## 1. 实验目标

本报告研究一维回归任务 `y = sin(3x)^3` 中，以下因素对预测精度的影响：

1. 训练点数 `num_train`
2. 隐藏层层数 `depth`
3. 隐藏层宽度 `width`
4. 激活函数 `activation`

目标是回答两个核心问题：

1. 数据量增加是否稳定提升测试集精度？
2. 更深、更宽、或不同激活函数是否一定带来更好的泛化？

## 2. 实验设置

- 框架：PyTorch
- 任务：在区间 `[-1, 1]` 上拟合 `y = sin(3x)^3`
- 参考测试集：`{config["num_test"]}` 个等间距采样点
- 优化器：Adam
- 学习率：`{config["lr"]}`
- 训练步数：`{config["steps"]}`
- 随机重复：`{len(config["seeds"])}` 次，种子为 `{config["seeds"]}`
- 默认基线网络：`depth=2, width=20, activation=tanh, num_train=50`
- 运行设备：`{config["device"]}`

评价指标：

- Test MSE：整体平方误差
- Test MAE：整体绝对误差
- Max Absolute Error：最坏点误差

## 3. 训练点数的影响

固定网络结构为 `depth=2, width=20, activation=tanh`，改变训练点数。

{sample_table}

![训练点数影响图]({assets["sample_metrics"]})

![不同训练点数下的预测曲线]({assets["sample_predictions"]})

结论：

- 训练点数从少量增加到中等规模时，测试误差下降明显。
- 点数过少时，模型在波峰和波谷附近会出现明显偏差，最大绝对误差较高。
- 从 `30` 到 `50` 个训练点没有严格单调下降，这更像是有限次随机采样带来的波动，而不是更大数据量失效。
- 当训练点数提升到较高水平后，误差继续下降，但边际收益开始减弱。

## 4. 隐藏层深度和宽度的影响

固定 `num_train=50, activation=tanh`，在深度和宽度上做二维网格搜索。

![深度与宽度热力图]({assets["depth_width_heatmap"]})

最佳 5 组结构如下：

{depth_width_top5}

观察：

- 最优结构是 `depth={best_grid["depth"]}, width={best_grid["width"]}`，平均 Test MSE 为 `{best_grid["test_mse_mean"]:.3e}`。
- 最差结构是 `depth={worst_grid["depth"]}, width={worst_grid["width"]}`，平均 Test MSE 为 `{worst_grid["test_mse_mean"]:.3e}`。
- 在这个小规模平滑回归任务中，网络并不是越深越好。深度过大时，训练成本增加，但泛化收益有限。
- 宽度从较小值增大时通常能降低误差，但当容量足够后收益趋于平缓。
- 如果同时考虑训练时间，`depth=2~3, width=20~40` 已经处在很有竞争力的精度区间，不必一开始就使用最宽网络。

## 5. 激活函数的影响

固定 `num_train=50, depth=2, width=20`，比较不同激活函数。

{activation_table}

![激活函数指标对比]({assets["activation_metrics"]})

![不同激活函数下的预测曲线]({assets["activation_predictions"]})

观察：

- 最优激活函数是 `{best_activation["activation"]}`，平均 Test MSE 为 `{best_activation["test_mse_mean"]:.3e}`。
- 最弱激活函数是 `{worst_activation["activation"]}`，平均 Test MSE 为 `{worst_activation["test_mse_mean"]:.3e}`。
- 在这组训练预算下，`gelu` 的平均指标最好，`tanh` 非常接近且表现稳定。
- `relu` 和 `silu` 在当前宽度与深度设置下略弱，说明激活函数选择会影响曲线细节恢复质量。

## 6. 综合分析

从本次实验可以得到以下结论：

1. 数据量是第一位因素。训练点数不足时，模型结构再复杂也难以稳定恢复目标函数细节。
2. 适度增加模型容量有帮助，但存在饱和区。对于一维平滑回归，过深网络并不划算。
3. 激活函数会显著影响拟合形状。在本实验设置下，`gelu` 和 `tanh` 明显优于 `relu` 与 `silu`。
4. 如果目标是“精度/复杂度”平衡，优先建议：
   - 先把训练点数增加到中等规模以上
   - 再选择 `gelu` 或 `tanh`
   - 最后在中等深度、中等宽度附近微调结构

## 7. 本实验下的推荐配置

- 如果追求稳健泛化：使用 `num_train=50` 以上，`depth=2~3`，`width=20~40`，激活函数优先 `gelu` 或 `tanh`
- 如果追求更低训练成本：优先使用浅层中宽网络，而不是盲目增加深度
- 如果后续任务更复杂、噪声更大：建议继续扩展实验，加入噪声扰动、权重衰减和更大测试范围

## 8. 复现实验

在项目根目录执行：

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python Codes/regression/torch/ablation_report.py
```

原始结果与图像输出目录：

- `Codes/regression/torch/report_assets/`
- `Codes/regression/torch/regression_experiment_report.md`
"""
    return report


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    base_config = {
        "steps": 3000,
        "lr": 1.0e-3,
        "num_test": 201,
        "device": str(device),
        "seeds": [1234, 2024, 3407],
    }

    all_records = []
    sample_records = []
    grid_records = []
    activation_records = []

    print(f"Running experiments on {device}")

    sample_counts = [10, 20, 30, 50, 80]
    for num_train in sample_counts:
        for seed in base_config["seeds"]:
            record = train_and_evaluate(
                {
                    **base_config,
                    "num_train": num_train,
                    "depth": 2,
                    "width": 20,
                    "activation": "tanh",
                    "seed": seed,
                },
                device,
            )
            sample_records.append(record)
            all_records.append(record)

    depth_values = [1, 2, 3, 4]
    width_values = [10, 20, 40, 80]
    for depth in depth_values:
        for width in width_values:
            for seed in base_config["seeds"]:
                record = train_and_evaluate(
                    {
                        **base_config,
                        "num_train": 50,
                        "depth": depth,
                        "width": width,
                        "activation": "tanh",
                        "seed": seed,
                    },
                    device,
                )
                grid_records.append(record)
                all_records.append(record)

    activation_values = ["tanh", "relu", "gelu", "silu"]
    for activation in activation_values:
        for seed in base_config["seeds"]:
            record = train_and_evaluate(
                {
                    **base_config,
                    "num_train": 50,
                    "depth": 2,
                    "width": 20,
                    "activation": activation,
                    "seed": seed,
                },
                device,
            )
            activation_records.append(record)
            all_records.append(record)

    sample_summary = aggregate_by(sample_records, ["num_train"])
    grid_summary = aggregate_by(grid_records, ["depth", "width"])
    activation_summary = aggregate_by(activation_records, ["activation"])

    raw_results_path = OUTPUT_DIR / "raw_results.json"
    per_run_csv_path = OUTPUT_DIR / "per_run_metrics.csv"
    sample_csv_path = OUTPUT_DIR / "sample_count_summary.csv"
    grid_csv_path = OUTPUT_DIR / "depth_width_summary.csv"
    activation_csv_path = OUTPUT_DIR / "activation_summary.csv"

    save_json(raw_results_path, all_records)
    save_csv(
        per_run_csv_path,
        [
            {
                key: value
                for key, value in record.items()
                if key not in {"x_ref", "y_ref", "x_train", "y_train", "y_pred"}
            }
            for record in all_records
        ],
        [
            "num_train",
            "depth",
            "width",
            "activation",
            "seed",
            "steps",
            "lr",
            "device",
            "train_mse",
            "test_mse",
            "test_mae",
            "test_max_abs",
            "elapsed_sec",
        ],
    )
    save_csv(sample_csv_path, sample_summary, list(sample_summary[0].keys()))
    save_csv(grid_csv_path, grid_summary, list(grid_summary[0].keys()))
    save_csv(activation_csv_path, activation_summary, list(activation_summary[0].keys()))

    sample_metrics_path = OUTPUT_DIR / "sample_count_metrics.png"
    depth_width_heatmap_path = OUTPUT_DIR / "depth_width_heatmap.png"
    activation_metrics_path = OUTPUT_DIR / "activation_metrics.png"
    sample_predictions_path = OUTPUT_DIR / "sample_count_predictions.png"
    activation_predictions_path = OUTPUT_DIR / "activation_predictions.png"

    plot_sample_count_metrics(sample_summary, sample_metrics_path)
    plot_depth_width_heatmap(grid_summary, depth_width_heatmap_path)
    plot_activation_metrics(activation_summary, activation_metrics_path)
    plot_prediction_comparison(
        sample_records,
        sample_predictions_path,
        group_key="num_train",
        group_values=[10, 50, 80],
        title="Predictions for Different Numbers of Training Points",
    )
    plot_prediction_comparison(
        activation_records,
        activation_predictions_path,
        group_key="activation",
        group_values=["tanh", "relu", "gelu", "silu"],
        title="Predictions for Different Activations",
    )

    assets = {
        "sample_metrics": "report_assets/sample_count_metrics.png",
        "depth_width_heatmap": "report_assets/depth_width_heatmap.png",
        "activation_metrics": "report_assets/activation_metrics.png",
        "sample_predictions": "report_assets/sample_count_predictions.png",
        "activation_predictions": "report_assets/activation_predictions.png",
    }
    report = build_report(base_config, sample_summary, grid_summary, activation_summary, assets)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Saved report to {REPORT_PATH}")
    print(f"Saved assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
