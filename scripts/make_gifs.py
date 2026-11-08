"""Training animations: all four methods side by side, one GIF per target.

  figures/gifs/training_<target>.gif

Uses the sample snapshots saved every `snapshot_every` iterations (seed 0).
A method that finished earlier (Gaussian VI uses fewer iterations) holds its
last frame.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from ivi.experiment import load_run  # noqa: E402
from ivi.plotting import INK_2, SHORT_LABELS, density_panel, setup_style  # noqa: E402
from ivi.targets import TARGETS, get_target  # noqa: E402
from ivi.utils import FIGURES_DIR  # noqa: E402

METHODS = ["gaussian_vi", "adversarial", "kivi", "sivi"]


def make_gif(name: str, fps: int = 8, max_frames: int = 60) -> Path:
    t = get_target(name)
    runs = {m: load_run(name, m, 0) for m in METHODS}
    end = max(int(r["snap_iters"][-1]) for r in runs.values())
    frame_iters = np.unique(np.linspace(1, end, max_frames).astype(int))

    fig, axes = plt.subplots(1, len(METHODS) + 1, figsize=(2.6 * (len(METHODS) + 1), 2.9))
    density_panel(axes[0], t, contours=False, title="True posterior")

    def draw(k):
        it = frame_iters[k]
        for ax, m in zip(axes[1:], METHODS):
            ax.clear()
            r = runs[m]
            idx = int(np.searchsorted(r["snap_iters"], it, side="right") - 1)
            idx = max(idx, 0)
            density_panel(ax, t, r["snap_samples"][idx], bins=70)
            ax.set_title(SHORT_LABELS[m], fontsize=10)
        fig.suptitle(f"{t.title}  ·  iteration {it}", fontsize=10.5, color=INK_2)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frame_iters), blit=False)
    out = FIGURES_DIR / "gifs" / f"training_{name}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main():
    setup_style()
    for name in (sys.argv[1:] or TARGETS):
        make_gif(name)


if __name__ == "__main__":
    main()
