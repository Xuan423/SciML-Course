import torch
import torch.nn as nn


def make_mlp(in_dim, out_dim, width, depth, activation=nn.Tanh):
    layers = []
    last = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(last, width))
        layers.append(activation())
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class CavityDeepONet(nn.Module):
    """DeepONet mapping a lid boundary profile to two velocity components."""

    def __init__(self, branch_dim=65, coord_dim=2, latent_dim=96, branch_width=128, trunk_width=128, depth=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.branch = make_mlp(branch_dim, 2 * latent_dim, branch_width, depth)
        self.trunk = make_mlp(coord_dim, 2 * latent_dim, trunk_width, depth)
        self.bias = nn.Parameter(torch.zeros(2))
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, branch_input, coords):
        b = self.branch(branch_input).view(branch_input.shape[0], 2, self.latent_dim)
        t = self.trunk(coords).view(coords.shape[0], 2, self.latent_dim)
        out = torch.einsum("bcl,pcl->bpc", b, t) / self.latent_dim**0.5
        return out + self.bias.view(1, 1, 2)
