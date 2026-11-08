"""Common interface for every VI method.

All methods minimise the same objective,

    KL(q || p(z|x)) = E_q[ log q(z) - log p~(z) ] + log Z,

and differ only in how they deal with the  E_q[log q(z)]  term (the negative
entropy). ``step`` performs one update of the variational parameters and returns
scalars to log. ``kl_est`` is the method's *own* running estimate of the KL
(its loss plus the known log Z), which we later compare with the true KL.
"""

from __future__ import annotations

import torch
from torch import nn

from ..targets import Target


class VIMethod(nn.Module):
    name = "base"
    label = "Base"
    has_exact_log_q = False  # True only for explicit families

    def __init__(self, target: Target, cfg: dict):
        super().__init__()
        self.target = target
        self.cfg = cfg
        self.batch_size = cfg["batch_size"]

    def step(self) -> dict:
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError

    def log_q_estimate(self, z: torch.Tensor) -> torch.Tensor | None:
        """The method's belief about log q(z): exact, estimated, or None."""
        return None
