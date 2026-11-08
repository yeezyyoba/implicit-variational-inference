# Implicit Variational Inference with Adversarial Density-Ratio Estimation — Summary

**Eyob Nebyou · iCog Labs AI training · November 2026** · code: `github.com/yeezyyoba/implicit-vi`

**Setup.** Three 2D posteriors p(z|x) ∝ p̃(z) = p(z)p(x|z): *Banana* (curved), *X-shape* (two crossing ridges), *Mixture* (4 separated modes). In 2D they can be normalised on a grid, so every fitted q is scored by its **true** reverse KL(q‖p), estimated without binning (kNN entropy + exact log p; it matches the closed form to 0.003 nats). All implicit methods use an 8-dim noise → 2D MLP, the same iteration budget (4k; 8k on the mixture) and 3 seeds.

**The intractable term.** VI minimises KL(q‖p) = E_q[log q(z) − log p̃(z)] + log Z. For an implicit q (z = G_φ(ε)), log q(z) cannot be evaluated. Each method handles it differently:

- **Gaussian VI (baseline):** q = N(μ, LLᵀ) has a closed-form log q. The objective is exact, but the family is only an ellipse.
- **Adversarial DRE (main):** add and subtract a tractable reference r: E_q[log q − log p̃] = E_q[log q/r] + E_q[log r − log p̃]. A classifier T trained with the logistic loss on q- vs r-samples has optimum σ(T\*) = q/(q+r), i.e. **T\*(z) = log q(z) − log r(z)**. The generator minimises E_ε[T(G(ε)) + log r − log p̃] with T frozen. That is the full gradient at T = T\*, because the φ-dependence of T\* contributes E_q[∇_φ log q_φ] = 0 (score identity). Hence the alternating loop: k classifier steps, then 1 generator step.
- **KIVI (bonus):** same decomposition, but q/r is fitted in closed form by kernel least squares, ρ̂ = Σα_j k(z, c_j) with α = (H + λI)⁻¹g. No inner loop and no game, but there is a bandwidth h and a ridge λ to choose, and ρ̂ can go negative (it is clipped).
- **SIVI (bonus):** no ratio at all. With q(z) = E_ψ[N(z; ψ, σ²)] and ψ = f_φ(ε), it maximises L_K = E[log p̃(z) − log (1/(K+1)) Σ_{k=0..K} q(z|ψ_k)] ≤ ELBO, which is monotone in K. K = 0 gives Gaussian VI.

**Results** (true KL(q‖p) in nats, mean ± sd over 3 seeds):

| | Gaussian VI | Adversarial DRE | KIVI | SIVI |
|---|---|---|---|---|
| Banana | 0.470 ± 0.004 | **0.007 ± 0.002** | 0.159 ± 0.076 | 0.013 ± 0.004 |
| X-shape | 0.368 ± 0.002 | 0.006 ± 0.005 | 0.269 ± 0.051 | **0.005 ± 0.002** |
| Mixture (modes) | 2.122 ± 0.266 (blob) | 0.044 ± 0.018 (4/4) | 1.396 ± 0.026 (1–2/4) | **0.020 ± 0.001** (4/4) |
| Time / run | 2 s | 73–180 s | 26–65 s | 32–72 s |

**Why the optimisation behaves as it does.**
1. *The classifier does learn the ratio.* Its logit tracks an independent KDE estimate of log q − log r (corr. 0.94 / 0.96). Its accuracy settles at the Bayes-optimal ½(1 + TV(p, r)) (0.892 vs 0.895 predicted on Banana), not at 50 %, because only the ratio matters.
2. *Gradients are biased whenever T ≠ T\*.* Its own KL estimate is noisy and can dip below 0. More inner steps help with diminishing returns (KL 0.010 / 0.008 / 0.003 for k = 1 / 5 / 10, at 28 / 65 / 199 s).
3. *Classifier lag gives tail leakage.* Far from q's samples, T extrapolates low, so moving mass there looks cheap. The tractable term is unbounded below wherever r has lighter tails than p (the banana's arms), so the ratio estimate alone must stop q escaping. A moving, moment-matched r (adaptive contrast) leaked 3.0 % of samples (KL 0.225) vs 0.085 % for the fixed prior (KL 0.008). Adaptive contrast pays off in high dimensions, not here. KIVI, whose kernels cannot extrapolate, needed a heavy-tailed Student-t r to stay stable.
4. *Reverse KL is mode-seeking, so modes are found one split at a time.* KL drops in steps log 4 → log 2 → 0.
5. *SIVI's K sets the bias.* The ψ₀-inclusive surrogate over-estimates log q, so spreading is under-rewarded. Banana KL is 0.477 / 0.140 / 0.037 / 0.013 for K = 0 / 10 / 50 / 500, and the mixture collapses to one mode for K ≤ 50 but finds all four at K ≥ 200. L_K → ELBO on a fixed q (Banana: −0.91 at K = 0 to 1.355 at K = 2000, true ELBO 1.355).
6. *KIVI moves the difficulty into hyperparameters.* The median-heuristic bandwidth feeds back on q's spread (h → 2.5, KL → 2.7). λ = 0.01 rescues the mixture but breaks the Banana (1.93).

**Trade-offs.** *Adversarial DRE:* the most flexible q and estimator. The cost is an inner loop, a biased and noisy gradient and KL estimate, and sensitivity to the choice of r. It scales to high dimensions with a neural ratio estimator. *KIVI:* deterministic and loop-free, but biased (smoothing and shrinkage), refit from scratch each step, hyperparameter-fragile, and kernel ratios degrade with dimension. *SIVI:* exact gradients of a valid bound, low variance, and its estimate is a conservative upper bound on KL. The cost is O(BK) per step, with small K under-dispersing, and q restricted to mixtures of explicit kernels. *Gaussian VI:* exact and cheapest, but structurally wrong for non-elliptical posteriors.
