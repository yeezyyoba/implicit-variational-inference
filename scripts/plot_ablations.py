"""Ablations of the adversarial method -> figures/06_ablations.png

(a) contrast distribution: fixed prior vs. adaptive (moment-matched Gaussian)
(b) classifier steps per generator step: k = 1, 5, 10 (Banana)
(c, d) where the adaptive reference goes wrong on the Banana: tail leakage

Requires:
  python scripts/run_all.py --methods adversarial --seeds 0
  python scripts/run_all.py --methods adversarial --seeds 0 --tag adaptive \
      --set adversarial.reference=adaptive
  python scripts/run_all.py --targets banana --methods adversarial --seeds 0 --tag k1 \
      --set adversarial.disc_steps=1
  python scripts/run_all.py --targets banana --methods adversarial --seeds 0 --tag k10 \
      --set adversarial.disc_steps=10
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from ivi.experiment import load_run  # noqa: E402
from ivi.plotting import (INK, INK_2, INK_3, METHOD_COLORS, density_panel, save,  # noqa: E402
                          setup_style)
from ivi.targets import TARGETS, get_target  # noqa: E402
from ivi.utils import FIGURES_DIR, RESULTS_DIR  # noqa: E402

BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
ORDINAL = {"k1": "#86b6ef", "": "#2a78d6", "k10": "#104281"}  # light -> dark = more steps


def main():
    setup_style()
    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.45, wspace=0.35)
    summary = {"reference": {}, "disc_steps": {}}

    # (a) reference distribution, all targets
    ax = fig.add_subplot(gs[0, :2])
    names = list(TARGETS)
    for i, name in enumerate(names):
        kp = load_run(name, "adversarial", 0)["metrics"]["kl_knn"]
        ka = load_run(name, "adversarial", 0, "adaptive")["metrics"]["kl_knn"]
        summary["reference"][name] = dict(prior=kp, adaptive=ka)
        y = len(names) - 1 - i
        ax.plot([max(kp, 1e-3), max(ka, 1e-3)], [y, y], color=INK_3, lw=1, zorder=1)
        ax.scatter([max(kp, 1e-3)], [y], s=70, color=METHOD_COLORS["adversarial"], zorder=3,
                   edgecolors="white", linewidths=1.5, label="r = prior (fixed)" if i == 0 else None)
        ax.scatter([max(ka, 1e-3)], [y], s=70, marker="D", color=INK_2, zorder=3,
                   edgecolors="white", linewidths=1.5,
                   label="r = adaptive (moment-matched Gaussian)" if i == 0 else None)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([get_target(n).title for n in names][::-1])
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 1)
    ax.set_xlabel("final true KL(q‖p)  (nats, log scale)")
    ax.grid(True, axis="x")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("(a) Contrast distribution r in the density ratio q/r", loc="left")

    # (b) classifier steps per generator step
    ax = fig.add_subplot(gs[0, 2:])
    for tag, k in (("k1", 1), ("", 5), ("k10", 10)):
        run = load_run("banana", "adversarial", 0, tag)
        h, m = run["history"], run["metrics"]
        x, y = np.array(h["eval_iter"]), np.maximum(np.array(h["kl_true"]), 1e-3)
        ax.plot(x, y, color=ORDINAL[tag], label=f"k = {k}:  final {m['kl_knn']:.3f},  "
                f"{m['train_time_s']:.0f} s")
        summary["disc_steps"][k] = dict(kl=m["kl_knn"], time_s=m["train_time_s"])
    ax.set_yscale("log")
    ax.set_xlabel("generator iteration")
    ax.set_ylabel("true KL(q‖p)")
    ax.grid(True)
    ax.legend(title="classifier steps per generator step", fontsize=8.5, title_fontsize=8.5)
    ax.set_title("(b) Inner-loop length (Banana)", loc="left")

    # (c, d) adaptive vs. prior samples on the banana, leaked samples highlighted
    t = get_target("banana")
    for j, (tag, title) in enumerate([("", "(c) r = prior"), ("adaptive", "(d) r = adaptive")]):
        ax = fig.add_subplot(gs[1, j])
        s = load_run("banana", "adversarial", 0, tag)["samples"]
        density_panel(ax, t, s)
        with torch.no_grad():
            lp = t.log_density(torch.as_tensor(s, dtype=torch.float64)).numpy()
        leak = lp < -8
        ax.scatter(s[leak, 0], s[leak, 1], s=5, color=METHOD_COLORS["adversarial"],
                   edgecolors="none", alpha=0.8)
        ax.set_title(title, loc="left")
        ax.text(0.03, 0.04, f"{100 * leak.mean():.1f}% of samples where\nlog p(z|x) < −8 (orange)",
                transform=ax.transAxes, fontsize=8.5, color=INK_2, bbox=BOX)
        summary["reference"].setdefault("banana_leak_frac", {})[tag or "prior"] = float(leak.mean())

    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    ax.text(0.0, 1.0,
            "Reading the ablations\n\n"
            "• The generator step treats T as fixed. That is exact only at the\n"
            "  optimal classifier T* (the φ-dependence of T* has zero expected\n"
            "  gradient), so a better-trained classifier means less biased\n"
            "  gradients: more inner steps help, with diminishing returns.\n\n"
            "• Far from q's samples the classifier extrapolates. Where it\n"
            "  under-estimates log q, moving mass there looks cheap, so the\n"
            "  generator leaks samples into the tails until T catches up.\n\n"
            "• A fixed broad prior keeps the classification task stationary\n"
            "  and covers q's tails. In 2D log q/prior is moderate, so the\n"
            "  classifier does not saturate. Adaptive contrast (AVB) is the\n"
            "  fix for high-dimensional z, where log q/prior is huge.",
            va="top", ha="left", fontsize=9.5, color=INK, linespacing=1.35)

    fig.suptitle("Ablations of adversarial implicit VI", fontsize=11.5, y=0.99)
    save(fig, FIGURES_DIR / "06_ablations.png")
    (RESULTS_DIR / "ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
