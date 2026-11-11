"""Plotting helpers with one consistent visual system.

* Densities (truth and fitted q) use one sequential blue ramp, light -> dark.
* The true posterior is overlaid on every fitted panel as thin ink contours at
  its 50 / 90 / 99 % highest-density levels, so "where q put mass vs. where p
  has mass" is readable at a glance.
* Methods keep a fixed categorical colour everywhere (lines, labels, legends).
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib

if "ipykernel" not in sys.modules:  # scripts save PNGs; notebooks keep inline display
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .targets import Target  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e8e7e2"

METHOD_COLORS = {
    "gaussian_vi": "#2a78d6",   # blue
    "adversarial": "#eb6834",   # orange
    "kivi": "#1baf7a",          # aqua
    "sivi": "#eda100",          # yellow
}
METHOD_LABELS = {
    "truth": "True posterior",
    "gaussian_vi": "Gaussian VI\n(explicit)",
    "adversarial": "Implicit VI\n(adversarial DRE)",
    "kivi": "KIVI\n(kernel DRE)",
    "sivi": "SIVI\n(semi-implicit)",
}
SHORT_LABELS = {"gaussian_vi": "Gaussian VI", "adversarial": "Adversarial DRE",
                "kivi": "KIVI", "sivi": "SIVI"}

SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelsize": 10, "axes.labelcolor": INK_2, "text.color": INK,
        "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0,
    })


def hpd_levels(density: np.ndarray, masses=(0.99, 0.9, 0.5)) -> list[float]:
    """Density thresholds enclosing the given probability masses (increasing)."""
    d = np.sort(density.ravel())[::-1]
    c = np.cumsum(d) / d.sum()
    return sorted(float(d[np.searchsorted(c, m)]) for m in masses)


def sample_density(samples: np.ndarray, target: Target, bins: int = 120) -> np.ndarray:
    (x0, x1), (y0, y1) = target.bounds
    H, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins,
                             range=[[x0, x1], [y0, y1]], density=True)
    return H.T


def density_panel(ax, target: Target, samples: np.ndarray | None = None,
                  bins: int = 120, contours: bool = True, vmax: float | None = None,
                  title: str | None = None, ticks: bool = False):
    """Heatmap of the truth (samples=None) or of q's samples, with truth contours."""
    gt = target.ground_truth(200)
    (x0, x1), (y0, y1) = target.bounds
    img = np.exp(gt["log_density"]) if samples is None else sample_density(samples, target, bins)
    vmax = vmax if vmax is not None else np.quantile(img, 0.995)
    ax.imshow(img, origin="lower", extent=(x0, x1, y0, y1), cmap=SEQ, vmin=0,
              vmax=max(vmax, 1e-12), aspect="auto", interpolation="bilinear")
    if contours:
        dens = np.exp(gt["log_density"])
        ax.contour(gt["X"], gt["Y"], dens, levels=hpd_levels(dens), colors=INK,
                   linewidths=0.6, alpha=0.75)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def label_line_end(ax, x, y, text, color, dy=0.0):
    """Direct label at the right end of a line (text in ink, not series colour)."""
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)
    ax.plot([x[-1]], [y[-1]], "o", ms=4, color=color)


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")
