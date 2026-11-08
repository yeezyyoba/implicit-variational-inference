"""Reference (contrast) distributions r(z) for density-ratio methods.

Both ratio-based methods write

    KL(q || p) - log Z = E_q[ log q/r ]  +  E_q[ log r - log p~ ]
                         (estimated)        (exact, given samples)

Any r with a tractable density that covers q is valid, but the choice matters:

* "prior"     N(0, s^2 I). Natural (prior-contrastive), fixed, broad.
              Caveat: if r's tails are *lighter* than the posterior's in some
              direction (true along the banana's arms), log r - log p~ -> -inf
              there. The objective then relies entirely on the ratio estimate
              to stop q escaping, and estimators are least accurate exactly
              where q has few samples.
* "student_t" multivariate t (df=3), same scale. Polynomial tails are heavier
              than any of our posteriors, so log r - log p~ -> +inf far away:
              escaping is never rewarded, whatever the estimator does there.
* "adaptive"  Gaussian moment-matched to the current q (AVB's adaptive
              contrast). Keeps q/r near 1, which is easiest to estimate, but r
              moves every step (non-stationary target for the estimator).
"""

from __future__ import annotations

import math

import torch
from torch.distributions import MultivariateNormal


class MultivariateStudentT:
    """Isotropic multivariate t with location 0, scale s, df nu (2D here)."""

    def __init__(self, scale: float, df: float = 3.0, dim: int = 2):
        self.scale, self.df, self.dim = scale, df, dim
        d, nu = dim, df
        self._log_norm = (math.lgamma((nu + d) / 2) - math.lgamma(nu / 2)
                          - 0.5 * d * math.log(nu * math.pi) - d * math.log(scale))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        m = (z ** 2).sum(-1) / self.scale ** 2
        return self._log_norm - 0.5 * (self.df + self.dim) * torch.log1p(m / self.df)

    def sample(self, shape) -> torch.Tensor:
        n = shape[0]
        g = torch.randn(n, self.dim)
        w = torch.distributions.Chi2(torch.tensor(float(self.df))).sample((n,))
        return self.scale * g / torch.sqrt(w / self.df).unsqueeze(-1)


@torch.no_grad()
def make_reference(kind: str, target, generate=None, ref_batch: int = 1024):
    """Build r(z). ``generate(n)`` (q's sampler) is only needed for 'adaptive'."""
    s = target.prior_std
    if kind == "prior":
        return MultivariateNormal(torch.zeros(2), scale_tril=s * torch.eye(2))
    if kind == "student_t":
        return MultivariateStudentT(scale=s, df=3.0)
    if kind == "adaptive":
        z = generate(ref_batch)
        cov = torch.cov(z.T) + 1e-3 * torch.eye(2)
        return MultivariateNormal(z.mean(0), covariance_matrix=cov)
    raise ValueError(f"unknown reference {kind!r}; use prior | student_t | adaptive")
