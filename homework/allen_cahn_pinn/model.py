import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(self, layers, activation: str = "tanh"):
        super().__init__()
        if len(layers) < 2:
            raise ValueError("`layers` must include input and output dimensions.")

        blocks = []
        act = _make_activation(activation)
        for i in range(len(layers) - 2):
            blocks.append(nn.Linear(layers[i], layers[i + 1]))
            blocks.append(act.__class__())
        blocks.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*blocks)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)
