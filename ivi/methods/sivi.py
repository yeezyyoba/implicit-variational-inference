"""Semi-Implicit Variational Inference (SIVI; Yin & Zhou, ICML 2018).

The family: an *explicit* conditional mixed over an *implicit* mixing distribution,

    psi = f_phi(eps),  eps ~ N(0, I)            (implicit: any neural net)
    z | psi ~ q(z | psi) = N(psi, diag(sigma^2)) (explicit: closed-form density)
    q(z) = E_psi[ q(z | psi) ]                   (intractable marginal)

Sampling is trivial, and q(z | psi) can be evaluated. Only the marginal q(z),
an expectation over psi, is unknown.

The bound (no density ratio at all)
----------------------------------
Replace log q(z) by the log of a Monte-Carlo mixture that *includes the psi
that generated z*:

    L_K = E_{psi^0, ..., psi^K ~ q(psi), z ~ q(z|psi^0)}
          [ log p~(z) - log( 1/(K+1) sum_{k=0}^K q(z | psi^k) ) ].

Facts (Yin & Zhou 2018, and the corrected bound with psi^0 included):
  * L_K <= ELBO for every K (it is a valid lower bound, so maximising it is safe),
  * L_K increases monotonically in K and L_K -> ELBO as K -> infinity.
  * K = 0 gives  E_psi[ ELBO of q(z|psi) ]: every psi is pushed to the same best
    Gaussian, so SIVI collapses to (diagonal) Gaussian VI. The K extra samples
    are what reward spreading psi out.
Including psi^0 matters: without it the mixture can miss the component that
produced z, log q is under-estimated, and the "bound" can exceed the ELBO, so the
optimiser could be rewarded for a degenerate q.

Everything is differentiated exactly by reparameterisation: no inner loop, no
classifier, no kernel. The price: the bound is loose for small K (bias towards
under-dispersion), the cost is O(B K) density evaluations per step, and q must
be a *mixture of explicit kernels*, which constrains the family's tails/shape.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..networks import MLP
from .base import VIMethod

LOG2PI = math.log(2 * math.pi)


def log_normal_diag(z: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """log N(z | mu, diag(exp(log_sigma)^2)), broadcasting over leading dims."""
    return (-0.5 * ((z - mu) / log_sigma.exp()) ** 2 - log_sigma - 0.5 * LOG2PI).sum(-1)


class SemiImplicitVI(VIMethod):
    name = "sivi"
    label = "SIVI (semi-implicit)"

    def __init__(self, target, cfg):
        super().__init__(target, cfg)
        self.noise_dim = cfg["noise_dim"]
        self.mixer = MLP(self.noise_dim, 2, cfg["hidden"], cfg["layers"])  # psi = f(eps)
        self.log_sigma = nn.Parameter(torch.full((2,), math.log(cfg.get("init_sigma", 0.3))))
        self.opt = torch.optim.Adam(self.parameters(), lr=cfg["lr"])
        self.K = cfg["K"]
        self.K_eval = cfg.get("K_eval", 2000)

    def psi(self, n: int) -> torch.Tensor:
        return self.mixer(torch.randn(n, self.noise_dim))

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        return self.psi(n) + self.log_sigma.exp() * torch.randn(n, 2)

    def surrogate_log_q(self, z: torch.Tensor, psi0: torch.Tensor, K: int) -> torch.Tensor:
        """log( 1/(K+1) [ q(z|psi0) + sum_k q(z|psi_k) ] ), psi_k shared across the batch."""
        own = log_normal_diag(z, psi0, self.log_sigma).unsqueeze(1)          # (B, 1)
        if K > 0:
            psik = self.psi(K)                                              # (K, 2)
            others = log_normal_diag(z.unsqueeze(1), psik.unsqueeze(0), self.log_sigma)
            comps = torch.cat([own, others], dim=1)                         # (B, K+1)
        else:
            comps = own
        return torch.logsumexp(comps, dim=1) - math.log(K + 1)

    def step(self) -> dict:
        psi0 = self.psi(self.batch_size)
        z = psi0 + self.log_sigma.exp() * torch.randn_like(psi0)           # z ~ q(z|psi0)
        log_q_hat = self.surrogate_log_q(z, psi0, self.K)
        loss = (log_q_hat - self.target.log_unnorm(z)).mean()              # = -L_K
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return dict(loss=loss.item(), kl_est=loss.item() + self.target.log_Z,
                    sigma=float(self.log_sigma.exp().mean()))

    @torch.no_grad()
    def bound(self, K: int, n: int = 5000) -> float:
        """Monte-Carlo value of L_K (a lower bound on the ELBO)."""
        psi0 = self.psi(n)
        z = psi0 + self.log_sigma.exp() * torch.randn_like(psi0)
        out = []
        for i in range(0, n, 1000):   # chunk to keep memory small for big K
            sl = slice(i, i + 1000)
            out.append(self.target.log_unnorm(z[sl]) - self.surrogate_log_q(z[sl], psi0[sl], K))
        return float(torch.cat(out).mean())

    @torch.no_grad()
    def log_q_estimate(self, z: torch.Tensor) -> torch.Tensor:
        """log q(z) ~ log mean_k q(z | psi_k) with K_eval mixing samples."""
        psik = self.psi(self.K_eval)
        out = []
        for i in range(0, len(z), 2000):
            lq = log_normal_diag(z[i:i + 2000].unsqueeze(1), psik.unsqueeze(0), self.log_sigma)
            out.append(torch.logsumexp(lq, dim=1) - math.log(self.K_eval))
        return torch.cat(out)
