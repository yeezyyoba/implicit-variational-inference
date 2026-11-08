r"""Implicit VI with adversarial density-ratio estimation (the main assignment).

The variational family
----------------------
An *implicit* distribution: a neural sampler

    eps ~ N(0, I_d),     z = G_phi(eps).

Sampling and reparameterised gradients are trivial, but q_phi(z) has no
closed form (G is not invertible, d != 2), so log q_phi(z) cannot be evaluated.

The intractable term
--------------------
    KL(q || p(z|x)) = E_q[ log q(z) - log p~(z) ] + log Z.

Pick any reference r(z) whose density we *can* evaluate and that covers q
(here: the prior, or a Gaussian moment-matched to q). Add and subtract log r:

    E_q[log q - log p~] = E_q[ log q(z)/r(z) ] + E_q[ log r(z) - log p~(z) ].
                          \_ intractable _/    \_______ tractable ______/

Only the log density *ratio* is unknown. Ratios can be learned from samples.

Density-ratio estimation by classification
------------------------------------------
Train a classifier T_psi(z) (logit) to tell samples of q (label 1) from samples
of r (label 0) with the logistic loss

    L(psi) = -E_q[ log sigmoid(T(z)) ] - E_r[ log(1 - sigmoid(T(z))) ].

For a fixed z the optimum is  sigmoid(T*) = q / (q + r),  i.e.

    T*(z) = log q(z) - log r(z).

So the trained discriminator's *logit* is the log density ratio we need, and

    loss_G(phi) = E_eps[ T(G(eps)) + log r(G(eps)) - log p~(G(eps)) ]

is a (biased, because T != T*) estimate of KL - log Z.

Why we can hold T fixed when differentiating w.r.t. phi
------------------------------------------------------
T* depends on phi through q_phi. Its total gradient is
    grad_phi E_q[T*] = E_eps[ grad_z T* . dG/dphi ] + E_q[ grad_phi T*(z) ]
and the second term is  E_q[ grad_phi log q_phi(z) ] = 0  (score-function
identity). So at the optimal discriminator, the pathwise gradient through z is
the *whole* gradient. This is why the algorithm alternates: re-fit T (several
steps) so that T ~ T*, then take one generator step with T frozen.

Adaptive contrast (Mescheder et al., 2017)
-----------------------------------------
Classifiers estimate log ratios well only when the two distributions overlap.
If r is the broad prior and q is a thin banana, log q/r is huge on q's support,
the classifier saturates (it wins with ~100% accuracy) and its logit stops
tracking the true ratio. Using an r that tracks q (a Gaussian with q's current
mean and covariance) keeps the ratio close to 1 and the estimate accurate,
while log r stays exactly computable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..networks import MLP, frozen
from .base import VIMethod
from .reference import make_reference


class AdversarialVI(VIMethod):
    name = "adversarial"
    label = "Implicit VI (adversarial DRE)"

    def __init__(self, target, cfg):
        super().__init__(target, cfg)
        self.noise_dim = cfg["noise_dim"]
        self.generator = MLP(self.noise_dim, 2, cfg["gen_hidden"], cfg["gen_layers"])
        self.discriminator = MLP(2, 1, cfg["disc_hidden"], cfg["disc_layers"])
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=cfg["lr_gen"],
                                      betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=cfg["lr_disc"],
                                      betas=(0.5, 0.999))
        self.disc_steps = cfg["disc_steps"]
        self.disc_warmup = cfg["disc_warmup"]
        self.reference_type = cfg["reference"]
        self.ref_batch = cfg.get("ref_batch", 2048)
        self._warmed_up = False

    # ---- sampling -------------------------------------------------------
    def generate(self, n: int) -> torch.Tensor:
        """Reparameterised samples z = G(eps); differentiable w.r.t. phi."""
        return self.generator(torch.randn(n, self.noise_dim))

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        return self.generate(n)

    # ---- reference distribution r(z) ------------------------------------
    @torch.no_grad()
    def reference(self):
        """The contrast distribution r(z); see ivi/methods/reference.py."""
        return make_reference(self.reference_type, self.target, self.generate, self.ref_batch)

    # ---- the two alternating updates ------------------------------------
    def discriminator_step(self, ref) -> tuple[float, float]:
        """One logistic-regression step: q-samples labelled 1, r-samples 0."""
        with torch.no_grad():
            zq = self.generate(self.batch_size)
            zr = ref.sample((self.batch_size,))
        logit_q = self.discriminator(zq).squeeze(-1)
        logit_r = self.discriminator(zr).squeeze(-1)
        # -log sigmoid(t) = softplus(-t);  -log(1 - sigmoid(t)) = softplus(t)
        loss = F.softplus(-logit_q).mean() + F.softplus(logit_r).mean()
        self.opt_d.zero_grad()
        loss.backward()
        self.opt_d.step()
        acc = 0.5 * ((logit_q > 0).float().mean() + (logit_r < 0).float().mean())
        return loss.item(), acc.item()

    def generator_step(self, ref) -> tuple[float, torch.Tensor]:
        """One step on the estimated KL with the ratio estimator T held fixed."""
        z = self.generate(self.batch_size)
        with frozen(self.discriminator):
            log_ratio = self.discriminator(z).squeeze(-1)        # ~ log q(z) - log r(z)
        loss = (log_ratio + ref.log_prob(z) - self.target.log_unnorm(z)).mean()
        self.opt_g.zero_grad()
        loss.backward()
        self.opt_g.step()
        return loss.item(), log_ratio.detach()

    def step(self) -> dict:
        ref = self.reference()  # same r for the classifier and the generator step
        n_disc = self.disc_steps if self._warmed_up else self.disc_warmup
        self._warmed_up = True
        for _ in range(n_disc):
            d_loss, d_acc = self.discriminator_step(ref)
        g_loss, log_ratio = self.generator_step(ref)
        return dict(loss=g_loss, kl_est=g_loss + self.target.log_Z,
                    disc_loss=d_loss, disc_acc=d_acc,
                    log_ratio_mean=log_ratio.mean().item())

    # ---- the method's own view of log q ---------------------------------
    @torch.no_grad()
    def log_ratio(self, z: torch.Tensor) -> torch.Tensor:
        """Estimated log q(z) - log r(z): the classifier's logit T(z)."""
        return self.discriminator(z).squeeze(-1)

    @torch.no_grad()
    def log_q_estimate(self, z: torch.Tensor) -> torch.Tensor:
        """log q(z) ~ T(z) + log r(z). Only meaningful where q has mass."""
        return self.log_ratio(z) + self.reference().log_prob(z)
