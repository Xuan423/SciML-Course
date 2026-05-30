from dataclasses import dataclass

import torch


def _require_grad(x: torch.Tensor) -> torch.Tensor:
    if x.requires_grad:
        return x
    return x.detach().clone().requires_grad_(True)


def pde_residual(model, x, t, diffusion=1.0e-4, create_graph=True):
    x = _require_grad(x)
    t = _require_grad(t)

    xt = torch.cat([x, t], dim=1)
    u = model(xt)
    one_u = torch.ones_like(u)

    u_t = torch.autograd.grad(u, t, grad_outputs=one_u, create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=one_u, create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    residual = u_t - diffusion * u_xx + 5.0 * (u ** 3 - u)
    return residual


def compute_losses(model, batch, diffusion=1.0e-4):
    x_f = batch["x_f"]
    t_f = batch["t_f"]
    x_ic = batch["x_ic"]
    t_ic = batch["t_ic"]
    u_ic = batch["u_ic"]
    x_lb = batch["x_lb"]
    t_lb = batch["t_lb"]
    u_lb = batch["u_lb"]
    x_rb = batch["x_rb"]
    t_rb = batch["t_rb"]
    u_rb = batch["u_rb"]

    f = pde_residual(model, x_f, t_f, diffusion=diffusion, create_graph=True)
    loss_f = torch.mean(f ** 2)

    u_ic_pred = model(torch.cat([x_ic, t_ic], dim=1))
    loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)

    u_lb_pred = model(torch.cat([x_lb, t_lb], dim=1))
    u_rb_pred = model(torch.cat([x_rb, t_rb], dim=1))
    loss_bc = 0.5 * (torch.mean((u_lb_pred - u_lb) ** 2) + torch.mean((u_rb_pred - u_rb) ** 2))

    return {"f": loss_f, "ic": loss_ic, "bc": loss_bc}


@dataclass
class AdaptiveWeightConfig:
    mode: str = "equal_weights"
    ema: float = 0.9
    min_value: float = 0.1
    max_value: float = 10.0
    update_every: int = 10


class AdaptiveWeights:
    def __init__(self, cfg: AdaptiveWeightConfig):
        self.cfg = cfg
        self.lmb = {"f": 1.0, "ic": 1.0, "bc": 1.0}

    def _normalize(self):
        mean_val = sum(self.lmb.values()) / 3.0
        if mean_val <= 0.0:
            self.lmb = {"f": 1.0, "ic": 1.0, "bc": 1.0}
            return
        for k in self.lmb:
            self.lmb[k] /= mean_val

    def update(self, losses, step):
        if self.cfg.mode == "equal_weights":
            return self.get()

        if step % max(self.cfg.update_every, 1) != 0:
            return self.get()

        vals = {k: float(v.detach().item()) for k, v in losses.items()}
        mean_loss = (vals["f"] + vals["ic"] + vals["bc"]) / 3.0
        eps = 1.0e-12
        for key in ["f", "ic", "bc"]:
            target = vals[key] / max(mean_loss, eps)
            target = max(self.cfg.min_value, min(self.cfg.max_value, target))
            self.lmb[key] = self.cfg.ema * self.lmb[key] + (1.0 - self.cfg.ema) * target
        self._normalize()
        return self.get()

    def get(self):
        return {k: float(v) for k, v in self.lmb.items()}
