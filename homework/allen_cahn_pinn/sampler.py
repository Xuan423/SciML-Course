import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Domain:
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0


class AllenCahnSampler:
    def __init__(self, domain: Domain, device: torch.device, seed: int = 1234):
        self.domain = domain
        self.device = device
        self.rng = np.random.default_rng(seed)

    def _to_tensor(self, array):
        return torch.tensor(array, dtype=torch.float32, device=self.device)

    def sample_interior(self, n_points: int, strategy: str = "random"):
        if strategy == "uniform":
            n_x = int(math.sqrt(n_points))
            n_t = int(math.ceil(n_points / max(n_x, 1)))
            x = np.linspace(self.domain.x_min, self.domain.x_max, n_x, dtype=np.float32)
            t = np.linspace(self.domain.t_min, self.domain.t_max, n_t, dtype=np.float32)
            xx, tt = np.meshgrid(x, t, indexing="xy")
            pts = np.stack([xx.reshape(-1), tt.reshape(-1)], axis=1)
            if pts.shape[0] > n_points:
                ids = self.rng.choice(pts.shape[0], n_points, replace=False)
                pts = pts[ids]
            x_f = pts[:, :1]
            t_f = pts[:, 1:]
            return self._to_tensor(x_f), self._to_tensor(t_f)

        if strategy == "random":
            x = self.rng.uniform(self.domain.x_min, self.domain.x_max, (n_points, 1)).astype(np.float32)
            t = self.rng.uniform(self.domain.t_min, self.domain.t_max, (n_points, 1)).astype(np.float32)
            return self._to_tensor(x), self._to_tensor(t)

        raise ValueError(f"Unknown sampling strategy: {strategy}")

    def sample_ic(self, n_points: int):
        x = self.rng.uniform(self.domain.x_min, self.domain.x_max, (n_points, 1)).astype(np.float32)
        t = np.zeros((n_points, 1), dtype=np.float32)
        u = (x ** 2) * np.cos(np.pi * x)
        return self._to_tensor(x), self._to_tensor(t), self._to_tensor(u.astype(np.float32))

    def sample_bc(self, n_points: int):
        t = self.rng.uniform(self.domain.t_min, self.domain.t_max, (n_points, 1)).astype(np.float32)
        x_l = np.full((n_points, 1), self.domain.x_min, dtype=np.float32)
        x_r = np.full((n_points, 1), self.domain.x_max, dtype=np.float32)
        u_l = -np.ones((n_points, 1), dtype=np.float32)
        u_r = -np.ones((n_points, 1), dtype=np.float32)
        return (
            self._to_tensor(x_l),
            self._to_tensor(t),
            self._to_tensor(u_l),
            self._to_tensor(x_r),
            self._to_tensor(t.copy()),
            self._to_tensor(u_r),
        )

    def sample_candidates(self, n_points: int):
        return self.sample_interior(n_points, strategy="random")

