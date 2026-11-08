"""Sample-based metrics against the exact 2D ground truth.

Implicit methods give samples but no density, so every metric here works from
samples alone. That keeps the comparison fair across explicit and implicit q.

* grid_divergences: bin q's samples on the same grid as the true posterior and
  compute reverse KL(q || p), forward KL(p || q) and total variation.
  Reverse KL is what VI minimises; forward KL punishes *missing* mass (modes).
* kl_knn: binning-free reverse KL(q || p) = -H[q] - E_q[log p(z|x)], with the
  entropy H[q] from a k-nearest-neighbour estimator and log p exact. This is the
  headline metric: it sees far-tail mass exactly, which a histogram cannot.
* mmd2: kernel two-sample distance between q's samples and exact posterior samples.
* mode_coverage: how q's mass is split across known modes (mixture target only).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.special import digamma

from .targets import Target


def knn_entropy(samples: np.ndarray, k: int = 5) -> float:
    """Kozachenko-Leonenko estimator of differential entropy (nats).

    H ~ psi(N) - psi(k) + log V_d + (d/N) sum_i log eps_i,
    with eps_i the distance from sample i to its k-th nearest neighbour and V_d
    the volume of the unit d-ball. Consistent, and very accurate in 2D.
    """
    x = np.asarray(samples, dtype=np.float64)
    n, d = x.shape
    eps = cKDTree(x).query(x, k=k + 1)[0][:, -1]  # k+1: the point itself is 0
    eps = np.maximum(eps, 1e-12)
    log_vd = (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1)
    return float(digamma(n) - digamma(k) + log_vd + d * np.mean(np.log(eps)))


def kl_knn(samples: np.ndarray, target: Target, k: int = 5, max_n: int = 50000) -> dict:
    """Reverse KL(q || p) from samples only: -H_knn[q] - E_q[log p(z | x)]."""
    x = np.asarray(samples[:max_n], dtype=np.float64)
    with torch.no_grad():
        log_p = target.log_density(torch.as_tensor(x)).numpy()
    h = knn_entropy(x, k)
    return dict(kl_knn=float(-h - log_p.mean()), entropy_knn=h,
                cross_entropy=float(-log_p.mean()))


def true_bin_probs(target: Target, bins: int, refine: int = 4) -> np.ndarray:
    """Probability mass of the true posterior in each of bins x bins cells ([y, x])."""
    gt = target.ground_truth(bins * refine)
    p = np.exp(gt["log_density"]) * gt["dx"] * gt["dy"]
    p = p.reshape(bins, refine, bins, refine).sum(axis=(1, 3))
    return gt["in_box_mass"] * p / p.sum()


def grid_divergences(samples: np.ndarray, target: Target, bins: int = 50,
                     alpha: float = 0.1, p_floor: float = 1e-10) -> dict:
    """Histogram-based divergences between q (samples) and the true posterior.

    alpha is a small pseudo-count so that log q is finite in empty bins; samples
    that leave the bounding box go to an extra "outside" bin whose true mass is
    computed exactly, so leaking mass is penalised rather than ignored. Binning
    still blurs tails; prefer ``kl_knn`` for reverse KL.
    """
    samples = np.asarray(samples)
    (x0, x1), (y0, y1) = target.bounds
    xe, ye = np.linspace(x0, x1, bins + 1), np.linspace(y0, y1, bins + 1)
    H, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[xe, ye])
    H = H.T  # -> [y, x], same layout as the truth
    n = len(samples)
    n_out = n - H.sum()

    p_in = true_bin_probs(target, bins).ravel()
    p = np.append(p_in, 1.0 - p_in.sum())  # true mass outside the box
    p = np.maximum(p, p_floor)
    p /= p.sum()
    counts = np.append(H.ravel(), n_out)
    q = (counts + alpha) / (n + alpha * counts.size)

    return dict(
        kl_qp=float(np.sum(q * (np.log(q) - np.log(p)))),
        kl_pq=float(np.sum(p * (np.log(p) - np.log(q)))),
        tv=float(0.5 * np.abs(p - q).sum()),
        out_mass=float(n_out / n),
    )


def mmd2(x: np.ndarray, y: np.ndarray, scales=(0.5, 1.0, 2.0), max_n: int = 3000,
         seed: int = 0) -> float:
    """Unbiased squared MMD with a sum of RBF kernels.

    The base bandwidth is the median pairwise distance of ``y`` (pass the *true*
    samples as y) so every method is measured with the same kernel.
    """
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), min(max_n, len(x)), replace=False)]
    y = y[rng.choice(len(y), min(max_n, len(y)), replace=False)]
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    dyy = torch.cdist(y, y) ** 2
    med = dyy[dyy > 0].median()
    dxx, dxy = torch.cdist(x, x) ** 2, torch.cdist(x, y) ** 2

    def k(d):
        return sum(torch.exp(-d / (2 * s * med)) for s in scales)

    m, n = len(x), len(y)
    kxx, kyy, kxy = k(dxx), k(dyy), k(dxy)
    term_x = (kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
    term_y = (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
    return float(term_x + term_y - 2 * kxy.mean())


def mode_coverage(samples: np.ndarray, target: Target, radius_sd: float = 3.0,
                  covered_frac: float = 0.05) -> dict | None:
    """Mass q assigns to each known mode (within radius_sd standard deviations)."""
    if target.modes is None:
        return None
    d = np.linalg.norm(samples[:, None, :] - target.modes[None], axis=-1)
    nearest = d.argmin(1)
    inside = d.min(1) < radius_sd * target.mode_std
    k = len(target.modes)
    weights = np.array([np.mean(inside & (nearest == j)) for j in range(k)])
    return dict(
        weights=weights.round(4).tolist(),
        n_covered=int((weights > covered_frac).sum()),
        n_modes=k,
        off_mode_mass=float(1 - inside.mean()),
    )
