"""Headline comparison of all four methods over 3 seeds.

  09_comparison_grid.png     fitted posteriors (seed 0) with mean ± std KL over seeds
  10_metrics_summary.png     reverse KL, forward KL and TV per seed, per method
  results/comparison_table.md / .csv / .json

Requires:  python scripts/run_all.py          (all targets x methods x seeds 0 1 2)
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from ivi.experiment import load_run  # noqa: E402
from ivi.plotting import (INK_2, INK_3, METHOD_COLORS, METHOD_LABELS, SHORT_LABELS,  # noqa: E402
                          density_panel, save, setup_style)
from ivi.targets import TARGETS, get_target  # noqa: E402
from ivi.utils import FIGURES_DIR, RESULTS_DIR  # noqa: E402

METHODS = ["gaussian_vi", "adversarial", "kivi", "sivi"]
SEEDS = [0, 1, 2]
BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88)


def collect():
    rows = {}
    for t in TARGETS:
        for m in METHODS:
            ms = [load_run(t, m, s)["metrics"] for s in SEEDS]
            r = dict(target=t, method=m)
            for key in ("kl_knn", "kl_pq", "tv", "mmd2", "train_time_s"):
                v = np.array([x[key] for x in ms], float)
                r[key] = v.tolist()
                r[key + "_mean"], r[key + "_std"] = float(v.mean()), float(v.std(ddof=1))
            if "mode_coverage" in ms[0]:
                r["modes_covered"] = [x["mode_coverage"]["n_covered"] for x in ms]
                r["off_mode_mass"] = [x["mode_coverage"]["off_mode_mass"] for x in ms]
            rows[(t, m)] = r
    return rows


def write_tables(rows):
    out_json = {f"{t}/{m}": r for (t, m), r in rows.items()}
    (RESULTS_DIR / "comparison.json").write_text(json.dumps(out_json, indent=2))
    with open(RESULTS_DIR / "comparison_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "method", "kl_qp_mean", "kl_qp_std", "kl_pq_mean", "kl_pq_std",
                    "tv_mean", "tv_std", "mmd2_mean", "train_time_s_mean", "modes_covered"])
        for (t, m), r in rows.items():
            w.writerow([t, m, f"{r['kl_knn_mean']:.4f}", f"{r['kl_knn_std']:.4f}",
                        f"{r['kl_pq_mean']:.4f}", f"{r['kl_pq_std']:.4f}",
                        f"{r['tv_mean']:.4f}", f"{r['tv_std']:.4f}", f"{r['mmd2_mean']:.4f}",
                        f"{r['train_time_s_mean']:.1f}", r.get("modes_covered", "")])
    lines = ["# Comparison over 3 seeds (mean ± std)", "",
             "KL(q‖p): binning-free kNN estimate of the reverse KL (the VI objective). "
             "KL(p‖q): forward KL on a 50×50 grid (punishes missing mass). "
             "TV: total variation on the same grid. Time: optimisation wall-clock on a "
             "4-core CPU VM. Mixture: a mode counts if q puts ≥5% of its mass within 3 sd of it; "
             "'mass between modes' exposes broad blobs that touch every mode without fitting any.",
             ""]
    for t in TARGETS:
        lines += [f"## {get_target(t).title}", "",
                  "| Method | KL(q‖p) | KL(p‖q) | TV | Time (s) |" +
                  (" Modes with ≥5% mass | Mass between modes |" if t == "gmm" else ""),
                  "|---|---|---|---|---|" + ("---|---|" if t == "gmm" else "")]
        for m in METHODS:
            r = rows[(t, m)]
            line = (f"| {SHORT_LABELS[m]} | {r['kl_knn_mean']:.3f} ± {r['kl_knn_std']:.3f} | "
                    f"{r['kl_pq_mean']:.3f} ± {r['kl_pq_std']:.3f} | "
                    f"{r['tv_mean']:.3f} ± {r['tv_std']:.3f} | {r['train_time_s_mean']:.0f} |")
            if t == "gmm":
                line += (" " + ", ".join(f"{c}/4" for c in r["modes_covered"]) + " | "
                         + ", ".join(f"{100 * o:.0f}%" for o in r["off_mode_mass"]) + " |")
            lines.append(line)
        lines.append("")
    (RESULTS_DIR / "comparison_table.md").write_text("\n".join(lines))
    print("\n".join(lines))


def fig_grid(rows):
    names = list(TARGETS)
    cols = ["truth"] + METHODS
    fig, axes = plt.subplots(len(names), len(cols), figsize=(3.0 * len(cols), 3.0 * len(names)))
    for i, name in enumerate(names):
        t = get_target(name)
        for j, col in enumerate(cols):
            ax = axes[i, j]
            if col == "truth":
                density_panel(ax, t, contours=False)
            else:
                density_panel(ax, t, load_run(name, col, 0)["samples"])
                r = rows[(name, col)]
                ax.text(0.03, 0.04, f"KL {r['kl_knn_mean']:.3f} ± {r['kl_knn_std']:.3f}",
                        transform=ax.transAxes, fontsize=8.5, color=INK_2, bbox=BOX)
            if i == 0:
                ax.set_title(METHOD_LABELS[col], fontsize=10.5)
        axes[i, 0].set_ylabel(t.title, color=INK_2, fontsize=11)
    fig.suptitle("All methods, all targets  ·  panels: seed 0  ·  KL(q‖p) in nats: mean ± std "
                 "over 3 seeds  ·  contours: true 50/90/99% HPD", fontsize=10.5, y=1.0)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "09_comparison_grid.png")


def fig_metrics(rows):
    names = list(TARGETS)
    metrics = [("kl_knn", "reverse KL(q‖p)"), ("kl_pq", "forward KL(p‖q)"),
               ("tv", "total variation")]
    fig, axes = plt.subplots(len(metrics), len(names), figsize=(4.2 * len(names), 2.5 * len(metrics)),
                             sharey=True)
    for a, (key, label) in enumerate(metrics):
        for b, name in enumerate(names):
            ax = axes[a, b]
            for k, m in enumerate(METHODS):
                r = rows[(name, m)]
                y = len(METHODS) - 1 - k
                vals = np.maximum(r[key], 1e-3)
                ax.scatter(vals, [y] * len(vals), s=22, color=METHOD_COLORS[m], alpha=0.55,
                           edgecolors="none")
                ax.scatter([max(r[key + "_mean"], 1e-3)], [y], s=90, marker="|",
                           color=METHOD_COLORS[m], linewidths=2.5)
            ax.set_xscale("log")
            ax.set_xlim(8e-4, 8)
            ax.grid(True, axis="x")
            ax.set_yticks(range(len(METHODS)))
            ax.set_yticklabels([SHORT_LABELS[m] for m in METHODS][::-1])
            if a == 0:
                ax.set_title(get_target(name).title)
            if b == 0:
                ax.set_ylabel(label, fontsize=10, color=INK_2, labelpad=8)
    fig.suptitle("Per-seed metrics (dots) and seed means (bars), log scale  ·  reverse KL is what VI "
                 "minimises; forward KL penalises missing mass  ·  values < 0.001 shown at 0.001",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "10_metrics_summary.png")


def main():
    setup_style()
    rows = collect()
    write_tables(rows)
    fig_grid(rows)
    fig_metrics(rows)


if __name__ == "__main__":
    main()
