"""Bonus-method mechanics: what controls SIVI and KIVI.

  07_sivi_bound.png   SIVI: the number K of extra mixing samples
                      (a) final KL vs. K on Banana and the mixture
                      (b) L_K tightening towards the true ELBO on a trained model
                      (c-e) mixture fits for K = 0, 50, 500
  08_kivi_sensitivity.png   KIVI: bandwidth / ridge / reference
                      (a, b) final KL over a bandwidth x ridge grid (Banana, X-shape)
                      (c, d) the median-heuristic feedback loop: bandwidth and KL

Requires (after the default runs):
  for K in 0 10 50 200: run_all.py --targets banana gmm --methods sivi --seeds 0 \
      --tag K$K --set sivi.K=$K snapshot_every=0
  run_all.py --targets banana x_shape --methods kivi --seeds 0 --tag h<h>_lam<l> \
      --set kivi.bandwidth=<h> kivi.lam=<l>       for h in 0.2 0.3 0.4, l in 0.001 0.01
  run_all.py --targets banana --methods kivi --seeds 0 --tag median --set kivi.bandwidth=null
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from ivi.experiment import load_model, load_run, run_dir  # noqa: E402
from ivi.plotting import (INK, INK_2, INK_3, METHOD_COLORS, SEQ, density_panel, save,  # noqa: E402
                          setup_style)
from ivi.targets import get_target  # noqa: E402
from ivi.utils import FIGURES_DIR, RESULTS_DIR  # noqa: E402

BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
KS = [0, 10, 50, 200, 500]  # 500 is the default run (no tag)


def sivi_run(target, K):
    return load_run(target, "sivi", 0, "" if K == 500 else f"K{K}")


def fig_sivi():
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.42, wspace=0.3)
    out = {"kl_vs_K": {}, "bound_vs_K": {}}

    # (a) final KL vs K
    ax = fig.add_subplot(gs[0, 0])
    shades = {"banana": "#2a78d6", "gmm": "#104281"}
    for name in ("banana", "gmm"):
        kls = [sivi_run(name, K)["metrics"]["kl_knn"] for K in KS]
        out["kl_vs_K"][name] = dict(zip(map(str, KS), kls))
        ax.plot(range(len(KS)), np.maximum(kls, 1e-3), "o-", color=shades[name],
                label=get_target(name).title)
        g = load_run(name, "gaussian_vi", 0)["metrics"]["kl_knn"]
        ax.axhline(g, color=shades[name], lw=0.8, ls=":")
    ax.text(len(KS) - 1, 0.47, "Gaussian VI (dotted)", ha="right", va="bottom", fontsize=8,
            color=INK_2)
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_yscale("log")
    ax.set_xlabel("K  (extra mixing samples in the bound)")
    ax.set_ylabel("final true KL(q‖p)")
    ax.grid(True)
    ax.legend(fontsize=8.5, loc="center left")
    ax.set_title("(a) Tighter bound, better posterior", loc="left")

    # (b) L_K on the trained default model vs. the true ELBO
    ax = fig.add_subplot(gs[0, 1:])
    for name in ("banana", "gmm"):
        model = load_model(name, "sivi", 0)
        t = get_target(name)
        elbo = t.log_Z - load_run(name, "sivi", 0)["metrics"]["kl_knn"]
        Ks = [0, 1, 5, 20, 100, 500, 2000]
        bounds = [model.bound(K, n=4000) for K in Ks]
        out["bound_vs_K"][name] = dict(K=Ks, L_K=bounds, elbo=elbo)
        x = np.arange(len(Ks))
        ax.plot(x, np.array(bounds) - elbo, "o-", color=shades[name], label=t.title)
    ax.axhline(0, color=INK, lw=1, ls="--")
    ax.text(len(Ks) - 1, 0.02, "true ELBO of the trained q  ", ha="right", va="bottom",
            fontsize=8.5, color=INK_2)
    ax.set_xticks(np.arange(len(Ks)))
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("K used to evaluate the bound")
    ax.set_ylabel("L_K − ELBO  (nats)")
    ax.grid(True)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("(b) L_K ≤ ELBO, increasing in K (same trained q, default K = 500)", loc="left")

    # (c-e) mixture fits at K = 0, 50, 500
    t = get_target("gmm")
    for j, K in enumerate([0, 50, 500]):
        ax = fig.add_subplot(gs[1, j])
        run = sivi_run("gmm", K)
        density_panel(ax, t, run["samples"])
        cov = run["metrics"]["mode_coverage"]
        ax.set_title(f"({'cde'[j]}) mixture, K = {K}", loc="left")
        ax.text(0.03, 0.04, f"KL = {run['metrics']['kl_knn']:.3f}\n"
                f"modes covered: {cov['n_covered']}/4", transform=ax.transAxes, fontsize=9,
                color=INK_2, bbox=BOX)
    fig.suptitle("SIVI: the bound's tightness K decides whether q can spread", fontsize=11.5,
                 y=0.995)
    save(fig, FIGURES_DIR / "07_sivi_bound.png")
    return out


def fig_kivi():
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28)
    hs, lams = [0.2, 0.3, 0.4], [0.001, 0.01]
    out = {"grid": {}}
    for j, name in enumerate(["banana", "x_shape"]):
        grid = np.full((len(lams), len(hs)), np.nan)
        for a, lam in enumerate(lams):
            for b, h in enumerate(hs):
                tag = "" if (h == 0.3 and lam == 0.001) else f"h{h}_lam{lam}"
                if (run_dir(name, "kivi", 0, tag) / "metrics.json").exists():
                    grid[a, b] = load_run(name, "kivi", 0, tag)["metrics"]["kl_knn"]
        out["grid"][name] = grid.tolist()
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(grid, cmap=SEQ, norm=LogNorm(vmin=0.1, vmax=3), aspect="auto")
        for a in range(len(lams)):
            for b in range(len(hs)):
                v = grid[a, b]
                ax.text(b, a, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                        fontsize=11, color="white" if v > 0.8 else INK)
        ax.set_xticks(range(len(hs)))
        ax.set_xticklabels([f"h = {h}" for h in hs])
        ax.set_yticks(range(len(lams)))
        ax.set_yticklabels([f"λ = {lam}" for lam in lams])
        g = load_run(name, "gaussian_vi", 0)["metrics"]["kl_knn"]
        ax.set_title(f"({'ab'[j]}) {get_target(name).title}: final KL "
                     f"(Gaussian VI: {g:.2f})", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=fig.axes[:2], fraction=0.025, pad=0.02)
    cb.set_label("true KL(q‖p), log scale", color=INK_2)
    cb.outline.set_visible(False)

    # (c, d) median heuristic feedback loop on the banana
    med = load_run("banana", "kivi", 0, "median")["history"]
    fix = load_run("banana", "kivi", 0)["history"]
    ax = fig.add_subplot(gs[1, 0])
    for h, lab, col in ((med, "median heuristic h = median|zᵢ − zⱼ|", INK_2),
                        (fix, "fixed h = 0.3", METHOD_COLORS["kivi"])):
        ax.plot(h["iter"], h["bandwidth"], color=col, label=lab)
    ax.set_xlabel("iteration")
    ax.set_ylabel("kernel bandwidth h")
    ax.grid(True)
    ax.legend(fontsize=8.5)
    ax.set_title("(c) Bandwidth tied to q's own spread grows…", loc="left")
    ax = fig.add_subplot(gs[1, 1])
    for h, lab, col in ((med, "median heuristic", INK_2), (fix, "fixed h = 0.3",
                                                            METHOD_COLORS["kivi"])):
        ax.plot(h["eval_iter"], np.maximum(h["kl_true"], 1e-3), color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("true KL(q‖p)")
    ax.grid(True)
    ax.legend(fontsize=8.5)
    ax.set_title("(d) …so the ratio is over-smoothed and q over-disperses", loc="left")
    out["median_final_kl"] = float(med["kl_true"][-1])
    out["median_final_bandwidth"] = float(med["bandwidth"][-1])
    fig.suptitle("KIVI: a closed-form ratio estimator moves the difficulty into its "
                 "hyperparameters (Banana tuned: h = 0.3, λ = 0.001)", fontsize=11.5, y=0.995)
    save(fig, FIGURES_DIR / "08_kivi_sensitivity.png")
    return out


def main():
    setup_style()
    out = dict(sivi=fig_sivi(), kivi=fig_kivi())
    (RESULTS_DIR / "bonus_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
