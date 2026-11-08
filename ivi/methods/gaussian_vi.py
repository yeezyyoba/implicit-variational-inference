"""Explicit baseline: full-covariance Gaussian VI.

q(z) = N(mu, L L^T) with L lower-triangular (positive diagonal).

Because q has a closed-form density, nothing is intractable:
    loss = E_q[ log q(z) - log p~(z) ]   (the negative ELBO)
is estimated by Monte Carlo with the reparameterisation z = mu + L eps, and
log q(z) is evaluated exactly. Low-variance, stable, cheap.

The price is expressiveness: a Gaussian is unimodal and elliptical, so it can
capture only linear correlation. Under reverse KL it is *mode-seeking*: it
shrinks to sit inside the posterior rather than spread over regions where the
posterior is tiny, so it under-estimates uncertainty.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import MultivariateNormal

from .base import VIMethod


class GaussianVI(VIMethod):
    name = "gaussian_vi"
    label = "Gaussian VI (explicit)"
    has_exact_log_q = True

    def __init__(self, target, cfg):
        super().__init__(target, cfg)
        # small random start (seeded): a mean of exactly 0 is a symmetric saddle
        # point on the mixture target and would hide the usual mode-seeking.
        self.mu = nn.Parameter(0.5 * torch.randn(2))
        # softplus(0.5413) = 1  -> start from N(0, I)
        self.raw_diag = nn.Parameter(torch.full((2,), math.log(math.e - 1)))
        self.offdiag = nn.Parameter(torch.zeros(1))
        self.register_buffer("e10", torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
        self.opt = torch.optim.Adam(self.parameters(), lr=cfg["lr"])

    def scale_tril(self) -> torch.Tensor:
        return torch.diag_embed(F.softplus(self.raw_diag) + 1e-4) + self.offdiag * self.e10

    def dist(self) -> MultivariateNormal:
        return MultivariateNormal(self.mu, scale_tril=self.scale_tril())

    def step(self) -> dict:
        q = self.dist()
        z = q.rsample((self.batch_size,))                       # reparameterised
        loss = (q.log_prob(z) - self.target.log_unnorm(z)).mean()  # -ELBO
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return dict(loss=loss.item(), kl_est=loss.item() + self.target.log_Z)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        return self.dist().sample((n,))

    def log_q_estimate(self, z):
        return self.dist().log_prob(z)
