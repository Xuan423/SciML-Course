import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def resolve_path(path_like, base_dir):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def relative_l2(pred, target, eps=1.0e-12):
    pred = np.asarray(pred)
    target = np.asarray(target)
    diff = pred.reshape(pred.shape[0], -1) - target.reshape(target.shape[0], -1)
    base = target.reshape(target.shape[0], -1)
    return np.linalg.norm(diff, axis=1) / (np.linalg.norm(base, axis=1) + eps)


def mse(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    return float(np.mean((pred - target) ** 2))


def compute_metrics(pred, target):
    rel_all = relative_l2(pred, target)
    rel_u = relative_l2(pred[..., :1], target[..., :1])
    rel_v = relative_l2(pred[..., 1:], target[..., 1:])
    per_sample = rel_all.tolist()
    return {
        "relative_l2_mean": float(np.mean(rel_all)),
        "relative_l2_std": float(np.std(rel_all)),
        "mse": mse(pred, target),
        "u_relative_l2_mean": float(np.mean(rel_u)),
        "u_mse": mse(pred[..., 0], target[..., 0]),
        "v_relative_l2_mean": float(np.mean(rel_v)),
        "v_mse": mse(pred[..., 1], target[..., 1]),
        "per_sample_relative_l2": per_sample,
    }


def save_json(path, data):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def history_to_rows(history, model_name):
    rows = []
    for item in history:
        row = {"model": model_name}
        row.update(item)
        rows.append(row)
    return rows
