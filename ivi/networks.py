"""Neural network building blocks shared by the implicit methods."""

from __future__ import annotations

import contextlib

import torch
from torch import nn


class MLP(nn.Module):
    """Plain fully connected network: in -> [hidden x n_layers] -> out."""

    def __init__(self, d_in: int, d_out: int, hidden: int = 128, n_layers: int = 3,
                 activation: type[nn.Module] = nn.SiLU):
        super().__init__()
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), activation()]
            d = hidden
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@contextlib.contextmanager
def frozen(module: nn.Module):
    """Temporarily stop gradients flowing *into* a module's parameters.

    Gradients still flow *through* the module to its inputs. Used to hold the
    density-ratio estimator fixed while updating the variational generator.
    """
    flags = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield module
    finally:
        for p, f in zip(module.parameters(), flags):
            p.requires_grad_(f)
