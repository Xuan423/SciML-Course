import numpy as np
import torch

from .utils import compute_metrics


@torch.no_grad()
def predict_deeponet(model, data, loader, device):
    model.eval()
    coords = data.normalized_coords_tensor(device)
    preds = []
    for bc, _ in loader:
        bc = bc.to(device)
        pred = model(bc, coords).cpu().numpy()
        preds.append(pred)
    pred = np.concatenate(preds, axis=0)
    h, w = data.grid_shape
    pred = pred.reshape(pred.shape[0], h, w, 2)
    return data.decode_output(pred)


@torch.no_grad()
def predict_fno(model, data, loader, device):
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        preds.append(pred)
    pred = np.concatenate(preds, axis=0)
    return data.decode_output(pred)


def evaluate_predictions(name, pred, target):
    metrics = compute_metrics(pred, target)
    flat_rows = []
    for idx, value in enumerate(metrics["per_sample_relative_l2"]):
        flat_rows.append({"model": name, "sample": idx, "relative_l2": float(value)})
    summary = {
        "model": name,
        "relative_l2_mean": metrics["relative_l2_mean"],
        "relative_l2_std": metrics["relative_l2_std"],
        "mse": metrics["mse"],
        "u_relative_l2_mean": metrics["u_relative_l2_mean"],
        "u_mse": metrics["u_mse"],
        "v_relative_l2_mean": metrics["v_relative_l2_mean"],
        "v_mse": metrics["v_mse"],
    }
    return summary, flat_rows, metrics
