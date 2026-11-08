"""Figure 1: the three target posteriors and what an explicit Gaussian can do.

Requires:  python scripts/run_all.py --methods gaussian_vi --seeds 0
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from ivi.experiment import load_run  # noqa: E402
from ivi.plotting import INK_2, density_panel, save, setup_style  # noqa: E402
from ivi.targets import TARGETS, get_target  # noqa: E402
from ivi.utils import FIGURES_DIR, RESULTS_DIR  # noqa: E402


def main():
    setup_style()
    names = list(TARGETS)
    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.2))
    summary = {}
    for j, name in enumerate(names):
        t = get_target(name)
        run = load_run(name, "gaussian_vi", 0)
        m = run["metrics"]
        density_panel(axes[0, j], t, contours=False, title=t.title)
        density_panel(axes[1, j], t, run["samples"])
        axes[1, j].text(0.03, 0.03, f"KL(q‖p) = {m['kl_exact']:.2f} nats",
                        transform=axes[1, j].transAxes, fontsize=9, color=INK_2)
        summary[name] = {k: m[k] for k in ("kl_exact", "kl_qp", "kl_pq", "tv", "mmd2")}
    axes[0, 0].set_ylabel("True posterior p(z | x)", color=INK_2)
    axes[1, 0].set_ylabel("Full-covariance\nGaussian VI q(z)", color=INK_2)
    for ax in axes[:, 0]:
        ax.yaxis.label.set_visible(True)
    fig.suptitle("The explicit baseline can only fit an ellipse "
                 "(contours: true 50/90/99% HPD regions)", y=1.0, fontsize=11)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "01_targets_and_gaussian_vi.png")
    out = RESULTS_DIR / "baseline_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
