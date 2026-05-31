from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset

from .utils import save_csv, save_json


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def encode(self, x):
        return (x - self.mean) / (self.std + 1.0e-6)

    def decode(self, x):
        return x * (self.std + 1.0e-6) + self.mean


def inspect_mat_file(path):
    data = sio.loadmat(path)
    rows = []
    for key, value in data.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        item = {
            "file": str(path),
            "key": key,
            "shape": str(tuple(arr.shape)),
            "dtype": str(arr.dtype),
            "min": float(np.min(arr)) if arr.size and np.issubdtype(arr.dtype, np.number) else "",
            "max": float(np.max(arr)) if arr.size and np.issubdtype(arr.dtype, np.number) else "",
            "mean": float(np.mean(arr)) if arr.size and np.issubdtype(arr.dtype, np.number) else "",
            "std": float(np.std(arr)) if arr.size and np.issubdtype(arr.dtype, np.number) else "",
        }
        rows.append(item)
    return rows


def load_cavity_mat(path):
    data = sio.loadmat(path)
    required = ["u_bc", "u_data", "v_data", "x_2d", "y_2d"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing fields in {path}: {missing}")
    u_bc = data["u_bc"].astype(np.float32)
    u_data = data["u_data"].astype(np.float32)
    v_data = data["v_data"].astype(np.float32)
    target = np.stack([u_data, v_data], axis=-1)
    return {
        "u_bc": u_bc,
        "target": target,
        "x_2d": data["x_2d"].astype(np.float32),
        "y_2d": data["y_2d"].astype(np.float32),
    }


class CavityDataModule:
    def __init__(self, train_path, test_path, output_dir):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.output_dir = Path(output_dir)
        self.train = load_cavity_mat(self.train_path)
        self.test = load_cavity_mat(self.test_path)
        self.x_2d = self.train["x_2d"]
        self.y_2d = self.train["y_2d"]
        self.coords = np.stack([self.x_2d.reshape(-1), self.y_2d.reshape(-1)], axis=-1).astype(np.float32)

        self.bc_normalizer = Normalizer(
            mean=np.mean(self.train["u_bc"], axis=0, keepdims=True).astype(np.float32),
            std=np.std(self.train["u_bc"], axis=0, keepdims=True).astype(np.float32),
        )
        self.out_normalizer = Normalizer(
            mean=np.mean(self.train["target"], axis=(0, 1, 2), keepdims=True).astype(np.float32),
            std=np.std(self.train["target"], axis=(0, 1, 2), keepdims=True).astype(np.float32),
        )

    @property
    def grid_shape(self):
        return self.x_2d.shape

    @property
    def n_points(self):
        return int(np.prod(self.grid_shape))

    def save_data_summary(self):
        rows = inspect_mat_file(self.train_path) + inspect_mat_file(self.test_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_csv(self.output_dir / "cavity_mat_summary.csv", rows)
        save_json(
            self.output_dir / "normalization_summary.json",
            {
                "bc_mean_shape": list(self.bc_normalizer.mean.shape),
                "bc_std_shape": list(self.bc_normalizer.std.shape),
                "output_mean": self.out_normalizer.mean.reshape(-1).tolist(),
                "output_std": self.out_normalizer.std.reshape(-1).tolist(),
                "coords_shape": list(self.coords.shape),
                "grid_shape": list(self.grid_shape),
            },
        )
        return rows

    def _normalized_targets(self, split):
        return self.out_normalizer.encode(split["target"]).astype(np.float32)

    def _normalized_bc(self, split):
        return self.bc_normalizer.encode(split["u_bc"]).astype(np.float32)

    def deeponet_loader(self, split_name, batch_size, shuffle):
        split = self.train if split_name == "train" else self.test
        bc = torch.tensor(self._normalized_bc(split), dtype=torch.float32)
        target = self._normalized_targets(split).reshape(split["target"].shape[0], -1, 2)
        target = torch.tensor(target, dtype=torch.float32)
        return DataLoader(TensorDataset(bc, target), batch_size=batch_size, shuffle=shuffle)

    def fno_loader(self, split_name, batch_size, shuffle):
        split = self.train if split_name == "train" else self.test
        x = self.make_fno_input(split)
        y = self._normalized_targets(split)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    def make_fno_input(self, split):
        bc = self._normalized_bc(split)
        n, width = bc.shape
        h, w = self.grid_shape
        if width != w:
            raise ValueError(f"Expected u_bc width {w}, got {width}")
        bc_field = np.repeat(bc[:, None, :], h, axis=1)
        x = np.repeat(self.x_2d[None, :, :], n, axis=0)
        y = np.repeat(self.y_2d[None, :, :], n, axis=0)
        return np.stack([bc_field, x, y], axis=-1).astype(np.float32)

    def decode_output(self, normalized):
        return self.out_normalizer.decode(normalized)

    def target(self, split_name):
        split = self.train if split_name == "train" else self.test
        return split["target"]

    def normalized_coords_tensor(self, device):
        return torch.tensor(self.coords, dtype=torch.float32, device=device)
