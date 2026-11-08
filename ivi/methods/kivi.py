"""Kernel Implicit Variational Inference (KIVI; Shi, Sun & Zhu, ICLR 2018).

Same implicit family and the same decomposition as the adversarial method,

    E_q[log q - log p~] = E_q[ log q(z)/r(z) ] + E_q[ log r(z) - log p~(z) ],

but the log density ratio is estimated by **closed-form kernel least squares**
instead of a trained neural classifier.

Ratio estimator (unconstrained least-squares importance fitting)
---------------------------------------------------------------
Model the ratio rho(z) = q(z)/r(z) as a kernel expansion around b samples of q,

    rho_hat(z) = sum_j alpha_j k(z, c_j),     k(z, c) = exp(-|z - c|^2 / 2h^2),

and minimise the squared error to the true ratio under r, plus a ridge penalty:

    1/2 E_r[(rho_hat - rho)^2] + lambda/2 |alpha|^2
  = 1/2 E_r[rho_hat^2] - E_q[rho_hat] + lambda/2 |alpha|^2 + const.

(the cross term E_r[rho_hat rho] = E_q[rho_hat] needs only q-samples!). With
H = (1/n_r) K_r^T K_r and g = (1/n_q) K_q^T 1 the minimiser is one linear solve:

    alpha = (H + lambda I)^{-1} g.

No inner optimisation loop, no adversarial game, deterministic given samples.
The costs move elsewhere: an O(b^3) solve per step, a bandwidth h and a ridge
lambda to choose, and a ratio estimate that can go negative (it is clipped).
Kernel estimators also degrade quickly with dimension, which is why KIVI is a
good fit for this 2D study but not a drop-in for high-dimensional latents.

Gradient
--------
As in the adversarial method, the fitted estimator is held fixed (stop-gradient
on alpha and the centres) and the pathwise gradient flows through z = G(eps)
into log rho_hat(z). The same score-function argument applies: at the true
ratio, its dependence on phi contributes zero expected gradient. The centres
and the evaluation points are *separate* q-batches, so the estimate is not
evaluated on the very points it was fitted to (which would bias log rho upward).
"""

from __future__ import annotations

import torch

from ..networks import MLP
from .base import VIMethod
from .reference import make_reference


def rbf(a: torch.Tensor, b: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.cdist(a, b) ** 2 / (2 * h ** 2))


class KernelImplicitVI(VIMethod):
    name = "kivi"
    label = "KIVI (kernel DRE)"

    def __init__(self, target, cfg):
        super().__init__(target, cfg)
        self.noise_dim = cfg["noise_dim"]
        self.generator = MLP(self.noise_dim, 2, cfg["gen_hidden"], cfg["gen_layers"])
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=cfg["lr_gen"],
                                      betas=(0.5, 0.999))
        self.n_basis = cfg["n_basis"]
        self.n_ref = cfg["n_ref"]
        self.lam = cfg["lam"]
        self.bw_scale = cfg["bw_scale"]
        self.bandwidth = cfg.get("bandwidth")  # fixed h; None -> median heuristic
        self.ratio_floor = cfg.get("ratio_floor", 1e-4)
        self.reference_type = cfg["reference"]
        self.ref_batch = cfg.get("ref_batch", 1024)
        self._fit = None

    def generate(self, n: int) -> torch.Tensor:
        return self.generator(torch.randn(n, self.noise_dim))

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        return self.generate(n)

    @torch.no_grad()
    def reference(self):
        """The contrast distribution r(z); see ivi/methods/reference.py."""
        return make_reference(self.reference_type, self.target, self.generate, self.ref_batch)

    # ---- closed-form ratio estimator ------------------------------------
    @torch.no_grad()
    def fit_ratio(self, ref) -> dict:
        centres = self.generate(self.n_basis)      # basis centres = q samples
        zr = ref.sample((self.n_ref,))
        if self.bandwidth is not None:
            h = torch.tensor(float(self.bandwidth))
        else:  # median heuristic on q's own pairwise distances (see module notes)
            d = torch.cdist(centres, centres)
            h = self.bw_scale * d[d > 0].median()
        Kc = rbf(centres, centres, h)              # k(q-samples, centres)
        Kr = rbf(zr, centres, h)                   # k(r-samples, centres)
        H = Kr.T @ Kr / self.n_ref
        g = Kc.mean(0)
        alpha = torch.linalg.solve(H + self.lam * torch.eye(self.n_basis), g)
        self._fit = dict(centres=centres, h=h, alpha=alpha,
                         ref_mean=float((Kr @ alpha).mean()))  # E_r[rho_hat], ~1 if good
        return self._fit

    def ratio(self, z: torch.Tensor, fit: dict | None = None) -> torch.Tensor:
        fit = fit or self._fit
        return rbf(z, fit["centres"], fit["h"]) @ fit["alpha"]

    def log_ratio(self, z: torch.Tensor, fit: dict | None = None) -> torch.Tensor:
        """Estimated log q(z) - log r(z) (ratio clipped below at ratio_floor)."""
        return torch.log(self.ratio(z, fit).clamp_min(self.ratio_floor))

    def step(self) -> dict:
        ref = self.reference()
        fit = self.fit_ratio(ref)
        z = self.generate(self.batch_size)          # fresh batch, not the centres
        rho = self.ratio(z, fit)
        log_ratio = torch.log(rho.clamp_min(self.ratio_floor))
        loss = (log_ratio + ref.log_prob(z) - self.target.log_unnorm(z)).mean()
        self.opt_g.zero_grad()
        loss.backward()
        self.opt_g.step()
        return dict(loss=loss.item(), kl_est=loss.item() + self.target.log_Z,
                    bandwidth=float(fit["h"]), clip_frac=float((rho < self.ratio_floor).float().mean()),
                    ratio_ref_mean=fit["ref_mean"])

    @torch.no_grad()
    def log_q_estimate(self, z: torch.Tensor) -> torch.Tensor:
        ref = self.reference()
        fit = self.fit_ratio(ref)
        return self.log_ratio(z, fit) + ref.log_prob(z)
