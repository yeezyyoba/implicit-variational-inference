"""Diagnostics that make the mechanics of density-ratio VI visible.

ratio_check
    Does the classifier really learn  log q(z) - log r(z) ?  We compare its
    logit T(z) with an *independent* estimate of the same quantity: a kernel
    density estimate of log q from fresh generator samples, minus the exact
    log r. Agreement on q's support is what licenses the whole method.

bayes_optimal_accuracy
    If q = p exactly, the best possible classifier between q and r has accuracy
    1/2 (1 + TV(p, r)). The discriminator's accuracy should settle there, not at
    50%: with a fixed prior reference, q and r stay distinguishable at the
    optimum, and that is fine; only their *ratio* matters.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import gaussian_kde

from .targets import Target


@torch.no_grad()
def ratio_check(method, n_kde: int = 10000, n_points: int = 3000, grid_n: int = 110,
                mask_frac: float = 2e-3, seed: int = 0) -> dict:
    """Learned log-ratio vs. KDE-based log q - log r, on a grid and at q-samples."""
    target: Target = method.target
    torch.manual_seed(seed)
    s = method.sample(n_kde).numpy().astype(np.float64)
    kde = gaussian_kde(s.T)
    ref = method.reference()

    (x0, x1), (y0, y1) = target.bounds
    X, Y = np.meshgrid(np.linspace(x0, x1, grid_n), np.linspace(y0, y1, grid_n))
    G = np.stack([X.ravel(), Y.ravel()], axis=1)
    q_grid = kde(G.T)
    Gt = torch.tensor(G, dtype=torch.float32)
    learned = method.log_ratio(Gt).numpy()
    kde_ratio = np.log(np.maximum(q_grid, 1e-300)) - ref.log_prob(Gt).numpy()
    mask = q_grid > mask_frac * q_grid.max()  # only where q has mass

    # at fresh q samples (where the generator's gradient is actually evaluated)
    zs = method.sample(n_points)
    at_learned = method.log_ratio(zs).numpy()
    at_kde = np.log(kde(zs.numpy().T.astype(np.float64))) - ref.log_prob(zs).numpy()
    err = at_learned - at_kde
    return dict(
        X=X, Y=Y,
        learned=learned.reshape(X.shape), kde_ratio=kde_ratio.reshape(X.shape),
        mask=mask.reshape(X.shape),
        points_learned=at_learned, points_kde=at_kde,
        corr=float(np.corrcoef(at_learned, at_kde)[0, 1]),
        bias=float(err.mean()), rmse=float(np.sqrt((err ** 2).mean())),
    )


def bayes_optimal_accuracy(target: Target, n: int = 400) -> float:
    """1/2 (1 + TV(p, r)) with r the prior: the classifier's ceiling when q = p."""
    gt = target.ground_truth(n)
    p = np.exp(gt["log_density"]) * gt["in_box_mass"]
    Z = torch.tensor(np.stack([gt["X"].ravel(), gt["Y"].ravel()], 1), dtype=torch.float64)
    r = np.exp(target.log_prior(Z).numpy()).reshape(p.shape)
    cell = gt["dx"] * gt["dy"]
    r_out = max(0.0, 1.0 - r.sum() * cell)
    p_out = max(0.0, 1.0 - gt["in_box_mass"])
    tv = 0.5 * (np.abs(p - r).sum() * cell + abs(r_out - p_out))
    return 0.5 * (1.0 + tv)
