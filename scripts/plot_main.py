"""Main-assignment figures: implicit VI with adversarial DRE vs. Gaussian VI.

  02_adversarial_vs_gaussian.png  fitted posteriors on all three targets
  03_density_ratio_check.png      does the classifier learn log q - log r?
  04_training_dynamics.png        true KL, the method's own KL estimate, and
                                  classifier accuracy vs. its Bayes-optimal value
  05_mode_discovery.png           how the implicit q splits onto separated modes

Requires:  python scripts/run_all.py --methods gaussian_vi adversarial --seeds 0
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from ivi.diagnostics import bayes_optimal_accuracy, ratio_check  # noqa: E402
from ivi.experiment import load_model, load_run  # noqa: E402
from ivi.plotting import (INK, INK_2, INK_3, METHOD_COLORS, METHOD_LABELS, SHORT_LABELS,  # noqa: E402
                          density_panel, save, setup_style)
from ivi.targets import TARGETS, get_target  # noqa: E402
from ivi.utils import FIGURES_DIR, RESULTS_DIR  # noqa: E402

DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#6da7ec", "#f0efec", "#f19a73", "#b8461b"])
BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)


def smooth(y, w=10):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def fig_comparison():
    names = list(TARGETS)
    cols = ["truth", "gaussian_vi", "adversarial"]
    fig, axes = plt.subplots(len(names), 3, figsize=(9.6, 3.0 * len(names)))
    summary = {}
    for i, name in enumerate(names):
        t = get_target(name)
        summary[name] = {}
        for j, col in enumerate(cols):
            ax = axes[i, j]
            if col == "truth":
                density_panel(ax, t, contours=False)
                continue
            m = load_run(name, col, 0)
            density_panel(ax, t, m["samples"])
            kl = m["metrics"]["kl_knn"]
            ax.text(0.03, 0.04, f"KL(q‖p) = {kl:.3f}", transform=ax.transAxes,
                    fontsize=9, color=INK_2, bbox=BOX)
            summary[name][col] = {k: m["metrics"][k] for k in ("kl_knn", "kl_pq", "tv", "mmd2")}
        axes[i, 0].set_ylabel(t.title, color=INK_2, fontsize=11)
    for j, col in enumerate(cols):
        axes[0, j].set_title(METHOD_LABELS[col])
    fig.suptitle("Implicit VI (adversarial density-ratio estimation) vs. explicit Gaussian VI\n"
                 "contours: true 50/90/99% HPD regions · KL(q‖p) in nats, binning-free kNN estimate",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "02_adversarial_vs_gaussian.png")
    return summary


def fig_ratio_check():
    names = ["banana", "x_shape"]
    fig = plt.figure(figsize=(11.5, 3.4 * len(names)))
    gs = fig.add_gridspec(len(names), 5, width_ratios=[1, 1, 0.05, 0.55, 1.1], wspace=0.08)
    axes = np.array([[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1]),
                      fig.add_subplot(gs[i, 4])] for i in range(len(names))])
    stats = {}
    for i, name in enumerate(names):
        t = get_target(name)
        method = load_model(name, "adversarial", 0)
        rc = ratio_check(method)
        stats[name] = {k: rc[k] for k in ("corr", "bias", "rmse")}
        (x0, x1), (y0, y1) = t.bounds
        vals = np.concatenate([rc["kde_ratio"][rc["mask"]], rc["learned"][rc["mask"]]])
        lim = np.quantile(np.abs(vals), 0.98)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        for j, (key, title) in enumerate([("kde_ratio", "log q(z) − log r(z)\n(KDE of q, exact r)"),
                                          ("learned", "classifier logit T(z)\n(learned)")]):
            ax = axes[i, j]
            img = np.where(rc["mask"], rc[key], np.nan)
            im = ax.imshow(img, origin="lower", extent=(x0, x1, y0, y1), cmap=DIVERGING,
                           norm=norm, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=10)
        axes[i, 0].set_ylabel(t.title, color=INK_2, fontsize=11)
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[i, 2]))
        cb.set_label("nats", color=INK_2)
        cb.outline.set_visible(False)

        ax = axes[i, 2]
        ax.scatter(rc["points_kde"], rc["points_learned"], s=4, alpha=0.25,
                   color=METHOD_COLORS["adversarial"], edgecolors="none")
        lo = np.quantile(np.r_[rc["points_kde"], rc["points_learned"]], 0.005)
        hi = np.quantile(np.r_[rc["points_kde"], rc["points_learned"]], 0.995)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("KDE log q − log r  at q-samples")
        ax.set_ylabel("learned T(z)")
        ax.text(0.04, 0.96, f"corr = {rc['corr']:.3f}\nbias = {rc['bias']:+.3f}\n"
                f"RMSE = {rc['rmse']:.3f}", transform=ax.transAxes, va="top", fontsize=9,
                color=INK_2, bbox=BOX)
        ax.grid(True)
        if i == 0:
            ax.set_title("agreement where the\ngenerator gradient is taken", fontsize=10)
    fig.suptitle("The discriminator's logit is the log density ratio  T*(z) = log q(z) − log r(z)"
                 "   (r = prior)", fontsize=11, y=1.01)
    save(fig, FIGURES_DIR / "03_density_ratio_check.png")
    return stats


def fig_dynamics():
    names = list(TARGETS)
    fig, axes = plt.subplots(2, len(names), figsize=(4.0 * len(names), 7.0))
    bayes = {}
    for j, name in enumerate(names):
        t = get_target(name)
        ax = axes[0, j]
        for meth in ("gaussian_vi", "adversarial"):
            h = load_run(name, meth, 0)["history"]
            x, y = np.array(h["eval_iter"]), np.maximum(np.array(h["kl_true"]), 1e-3)
            ax.plot(x, y, color=METHOD_COLORS[meth], label=f"{SHORT_LABELS[meth]}: true KL")
        h = load_run(name, "adversarial", 0)["history"]
        it, est = np.array(h["iter"]), np.array(h["kl_est"])
        w = 10
        ax.plot(it[w - 1:], np.maximum(smooth(est, w), 1e-3), color=METHOD_COLORS["adversarial"],
                lw=1.2, ls=":", label="Adversarial: its own KL estimate")
        ax.set_yscale("log")
        ax.set_title(t.title)
        ax.set_xlabel("iteration")
        ax.grid(True, which="major")
        if j == 0:
            ax.set_ylabel("KL(q‖p)  (nats, log scale)")
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[1, j]
        acc = np.array(h["disc_acc"])
        ax.plot(it[w - 1:], smooth(acc, w), color=METHOD_COLORS["adversarial"], lw=1.5)
        b = bayes_optimal_accuracy(t)
        bayes[name] = b
        ax.axhline(b, color=INK, lw=1, ls="--")
        ax.axhline(0.5, color=INK_3, lw=0.8, ls=":")
        ax.text(it[-1], b, "  Bayes-optimal\n  if q = p", va="center", ha="right", fontsize=8,
                color=INK_2, bbox=BOX)
        ax.set_ylim(0.45, 1.0)
        ax.set_xlabel("iteration")
        ax.grid(True)
        if j == 0:
            ax.set_ylabel("classifier accuracy (q vs. prior)")
    fig.suptitle("Training dynamics of adversarial implicit VI", fontsize=11, y=1.0)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "04_training_dynamics.png")
    return bayes


def fig_mode_discovery():
    t = get_target("gmm")
    run = load_run("gmm", "adversarial", 0)
    iters, snaps = run["snap_iters"], run["snap_samples"]
    picks = [500, 2000, 4000, 5000, 6000, 8000]
    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(2, len(picks), height_ratios=[1.15, 1], hspace=0.22)
    h = run["history"]
    x, y = np.array(h["eval_iter"]), np.array(h["kl_true"])
    for k, p in enumerate(picks):
        idx = int(np.argmin(np.abs(iters - p)))
        ax = fig.add_subplot(gs[0, k])
        density_panel(ax, t, snaps[idx], bins=80, vmax=0.35)
        ax.set_title(f"iteration {iters[idx]}", fontsize=10)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(x, y, color=METHOD_COLORS["adversarial"])
    for v, lab in [(np.log(4), "log 4: one mode"), (np.log(2), "log 2: two modes")]:
        ax.axhline(v, color=INK_3, lw=0.8, ls="--")
        ax.text(x[-1], v, lab + "  ", ha="right", va="bottom", fontsize=8.5, color=INK_2)
    for p in picks:
        ax.axvline(p, color=INK_3, lw=0.6, ls=":")
    ax.set_xlabel("iteration")
    ax.set_ylabel("true KL(q‖p)")
    ax.set_ylim(0, max(1.6, y.max() * 1.05))
    ax.grid(True)
    fig.suptitle("Mode discovery: reverse KL drops in steps as the implicit q splits onto new modes",
                 fontsize=11, y=0.99)
    save(fig, FIGURES_DIR / "05_mode_discovery.png")


def main():
    setup_style()
    out = dict(comparison=fig_comparison(), ratio_check=fig_ratio_check(),
               bayes_optimal_accuracy=fig_dynamics())
    fig_mode_discovery()
    (RESULTS_DIR / "main_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
