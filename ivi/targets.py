"""2D target posteriors with exact (grid-based) ground truth.

Each target is an *unnormalised* posterior  p~(z) = p(z) * p(x | z)  over z in R^2.
VI only ever needs  log p~(z)  (the normaliser log Z cancels out of every gradient),
but because the problem is 2D we can also normalise it numerically on a fine grid.
That gives us something real papers rarely have: the exact posterior, its log
evidence log Z, and therefore the *exact* KL of any fitted q to the truth.

Prior-contrastive framing
-------------------------
Every target is written as prior x likelihood with a broad Gaussian prior
p(z) = N(0, prior_std^2 I). The "likelihood" is simply  log p~(z) - log p(z).
This lets the adversarial method use the prior as its contrast (reference)
distribution, exactly as in Huszar (2017) / Mescheder et al. (2017).
"""

from __future__ import annotations

import math

import numpy as np
import torch

LOG2PI = math.log(2.0 * math.pi)


def _logsumexp_np(a: np.ndarray) -> float:
    m = a.max()
    return float(m + np.log(np.exp(a - m).sum()))


class Target:
    """Base class. Subclasses implement ``log_unnorm`` and set ``bounds``."""

    name = "base"
    title = "Base"
    bounds = ((-5.0, 5.0), (-5.0, 5.0))  # ((xmin, xmax), (ymin, ymax)), holds ~all mass
    prior_std = 3.0
    modes: np.ndarray | None = None       # only for multimodal targets
    mode_std: float | None = None

    def __init__(self) -> None:
        self._gt_cache: dict[int, dict] = {}

    # ---- densities --------------------------------------------------------
    def log_unnorm(self, z: torch.Tensor) -> torch.Tensor:
        """log p~(z) up to an additive constant. z: (..., 2) -> (...)."""
        raise NotImplementedError

    def log_prior(self, z: torch.Tensor) -> torch.Tensor:
        s = self.prior_std
        d = z.shape[-1]
        return -0.5 * (z ** 2).sum(-1) / s ** 2 - d * (math.log(s) + 0.5 * LOG2PI)

    def log_lik(self, z: torch.Tensor) -> torch.Tensor:
        return self.log_unnorm(z) - self.log_prior(z)

    # ---- ground truth -----------------------------------------------------
    def ground_truth(self, n: int = 400) -> dict:
        """Normalise the target on an n x n grid of cell centres.

        Returns a dict with the grid (xs, ys, X, Y), the normalised log density
        at each cell centre (array indexed [y, x]), cell sizes, and log Z.
        """
        if n in self._gt_cache:
            return self._gt_cache[n]
        (x0, x1), (y0, y1) = self.bounds
        dx, dy = (x1 - x0) / n, (y1 - y0) / n
        xs = x0 + (np.arange(n) + 0.5) * dx
        ys = y0 + (np.arange(n) + 0.5) * dy
        X, Y = np.meshgrid(xs, ys)  # [y, x] indexing
        Z = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float64)
        with torch.no_grad():
            lu = self.log_unnorm(Z).numpy().reshape(n, n)
        log_Z = _logsumexp_np(lu) + math.log(dx * dy)
        gt = dict(xs=xs, ys=ys, X=X, Y=Y, dx=dx, dy=dy,
                  log_density=lu - log_Z, log_Z=log_Z)
        self._gt_cache[n] = gt
        return gt

    @property
    def log_Z(self) -> float:
        """log of the normalising constant of p~ (the log evidence, log p(x))."""
        return self.ground_truth()["log_Z"]

    def sample_truth(self, n: int, seed: int = 0) -> np.ndarray:
        """Exact-up-to-grid-resolution samples: pick a cell by mass, jitter inside it."""
        gt = self.ground_truth()
        rng = np.random.default_rng(seed)
        p = np.exp(gt["log_density"]).ravel()
        p /= p.sum()
        idx = rng.choice(p.size, size=n, p=p)
        x = gt["X"].ravel()[idx] + rng.uniform(-0.5, 0.5, n) * gt["dx"]
        y = gt["Y"].ravel()[idx] + rng.uniform(-0.5, 0.5, n) * gt["dy"]
        return np.stack([x, y], axis=1)


class Banana(Target):
    """Curved, strongly non-linear dependence: z2 | z1 ~ N(b z1^2 - c, s^2).

    A Gaussian can only capture its linear correlation (which is ~zero!), so
    mean-field and full-covariance Gaussians both miss the curvature.
    """

    name = "banana"
    title = "Banana"
    bounds = ((-4.5, 4.5), (-2.5, 7.5))
    a, b, s = 1.25, 0.5, 0.5

    def log_unnorm(self, z):
        z1, z2 = z[..., 0], z[..., 1]
        c = self.b * self.a ** 2  # centres z2 so that E[z2] = 0
        return -0.5 * (z1 / self.a) ** 2 - 0.5 * ((z2 - self.b * z1 ** 2 + c) / self.s) ** 2


class XShape(Target):
    """Equal mixture of two strongly correlated Gaussians with opposite correlation.

    Mass lies on two crossing ridges. Its covariance is (almost) isotropic, so a
    moment-matched Gaussian is a round blob that puts mass in the empty wedges.
    """

    name = "x_shape"
    title = "X-shape"
    bounds = ((-5.0, 5.0), (-5.0, 5.0))
    var, cov = 2.0, 1.8

    def log_unnorm(self, z):
        z1, z2 = z[..., 0], z[..., 1]
        det = self.var ** 2 - self.cov ** 2
        q_pos = (self.var * z1 ** 2 - 2 * self.cov * z1 * z2 + self.var * z2 ** 2) / det
        q_neg = (self.var * z1 ** 2 + 2 * self.cov * z1 * z2 + self.var * z2 ** 2) / det
        return torch.logaddexp(-0.5 * q_pos, -0.5 * q_neg)


class GaussianMixture(Target):
    """Four well-separated, equally weighted modes. The test of mode coverage."""

    name = "gmm"
    title = "Gaussian mixture"
    bounds = ((-5.0, 5.0), (-5.0, 5.0))
    modes = np.array([[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])
    mode_std = 0.6

    def log_unnorm(self, z):
        m = torch.as_tensor(self.modes, dtype=z.dtype)
        d2 = ((z.unsqueeze(-2) - m) ** 2).sum(-1)  # (..., K)
        return torch.logsumexp(-0.5 * d2 / self.mode_std ** 2, dim=-1)


TARGETS = {cls.name: cls for cls in (Banana, XShape, GaussianMixture)}


def get_target(name: str) -> Target:
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}; choose from {list(TARGETS)}")
    return TARGETS[name]()
