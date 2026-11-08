"""Training loop, evaluation and per-run caching.

A *run* is one (target, method, seed, tag). Its outputs are written to
    results/runs/<target>/<method>[__<tag>]/seed<k>/
        metrics.json   final metrics + config + timings      (committed)
        history.json   training curves                       (committed)
        samples.npz    final samples + training snapshots    (git-ignored)
        model.pt       trained parameters                    (git-ignored)
Runs are skipped if metrics.json already exists, so long sweeps are resumable.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .methods import build_method
from .metrics import grid_divergences, mmd2, mode_coverage
from .targets import get_target
from .utils import RESULTS_DIR, method_config, set_seed

RUNS_DIR = RESULTS_DIR / "runs"


def run_dir(target: str, method: str, seed: int, tag: str = "") -> Path:
    folder = method + (f"__{tag}" if tag else "")
    return RUNS_DIR / target / folder / f"seed{seed}"


def train(method, n_iters: int, log_every: int = 20, eval_every: int = 200,
          n_eval: int = 20000, snapshot_every: int | None = None,
          n_snapshot: int = 3000, verbose: bool = False):
    """Run ``n_iters`` optimisation steps; return (history, snapshots)."""
    target = method.target
    hist = defaultdict(list)
    snapshots = []
    step_time = 0.0
    for it in range(1, n_iters + 1):
        t0 = time.perf_counter()
        logs = method.step()
        step_time += time.perf_counter() - t0

        if it == 1 or it % log_every == 0:
            hist["iter"].append(it)
            for k, v in logs.items():
                hist[k].append(float(v))
        if it % eval_every == 0 or it == n_iters:
            g = grid_divergences(method.sample(n_eval).numpy(), target)
            hist["eval_iter"].append(it)
            hist["kl_grid"].append(g["kl_qp"])
            hist["tv_grid"].append(g["tv"])
            if verbose:
                print(f"  it {it:5d}  loss {logs['loss']:+.3f}  KL_est {logs['kl_est']:.3f}"
                      f"  KL_grid {g['kl_qp']:.3f}  TV {g['tv']:.3f}", flush=True)
        if snapshot_every and (it == 1 or it % snapshot_every == 0):
            snapshots.append((it, method.sample(n_snapshot).numpy().astype(np.float32)))
    hist["train_time_s"] = step_time
    return dict(hist), snapshots


def evaluate(method, n_samples: int = 100000, seed: int = 0) -> tuple[dict, np.ndarray]:
    """Final metrics from samples, plus the method's own KL estimate."""
    target = method.target
    samples = method.sample(n_samples).numpy()
    truth = target.sample_truth(20000, seed=12345)
    out = dict(grid_divergences(samples, target))
    out["mmd2"] = mmd2(samples, truth, seed=seed)
    cov = mode_coverage(samples, target)
    if cov is not None:
        out["mode_coverage"] = cov

    # The method's *own* view of KL(q||p): uses its exact / estimated log q.
    z = torch.as_tensor(samples[:20000])
    with torch.no_grad():
        log_q = method.log_q_estimate(z)
        if log_q is not None:
            gap = (log_q - target.log_unnorm(z)).mean().item()
            key = "kl_exact" if method.has_exact_log_q else "kl_self_estimate"
            out[key] = gap + target.log_Z
    out["sample_mean"] = samples.mean(0).tolist()
    out["sample_cov"] = np.cov(samples.T).tolist()
    return out, samples


def run_one(target_name: str, method_name: str, seed: int = 0, tag: str = "",
            overrides: dict | None = None, force: bool = False,
            verbose: bool = True) -> dict:
    out_dir = run_dir(target_name, method_name, seed, tag)
    if (out_dir / "metrics.json").exists() and not force:
        if verbose:
            print(f"[skip] {out_dir.relative_to(RESULTS_DIR.parent)} exists")
        return json.loads((out_dir / "metrics.json").read_text())

    cfg = method_config(method_name, overrides)
    set_seed(seed)
    target = get_target(target_name)
    method = build_method(method_name, target, cfg)
    if verbose:
        print(f"[run] target={target_name} method={method_name} seed={seed} tag={tag or '-'}",
              flush=True)

    hist, snaps = train(
        method, cfg["n_iters"], log_every=cfg["log_every"], eval_every=cfg["eval_every"],
        n_eval=cfg["n_eval_samples"], snapshot_every=cfg.get("snapshot_every"),
        n_snapshot=cfg.get("n_snapshot_samples", 3000), verbose=verbose)
    metrics, samples = evaluate(method, cfg["n_final_samples"], seed=seed)
    metrics.update(target=target_name, method=method_name, seed=seed, tag=tag,
                   log_Z=target.log_Z, train_time_s=hist["train_time_s"], config=cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(hist))
    snap_iters = np.array([s[0] for s in snaps])
    snap_samples = np.stack([s[1] for s in snaps]) if snaps else np.zeros((0, 0, 2))
    np.savez_compressed(out_dir / "samples.npz", samples=samples[:20000].astype(np.float32),
                        snap_iters=snap_iters, snap_samples=snap_samples)
    torch.save(method.state_dict(), out_dir / "model.pt")
    if verbose:
        print(f"[done] KL(q||p)={metrics['kl_qp']:.3f}  KL(p||q)={metrics['kl_pq']:.3f}  "
              f"TV={metrics['tv']:.3f}  MMD2={metrics['mmd2']:.4f}  "
              f"time={hist['train_time_s']:.1f}s", flush=True)
    return metrics


def load_run(target: str, method: str, seed: int = 0, tag: str = "") -> dict:
    """Load a cached run: metrics, history and (if present) samples."""
    d = run_dir(target, method, seed, tag)
    out = dict(metrics=json.loads((d / "metrics.json").read_text()),
               history=json.loads((d / "history.json").read_text()), dir=d)
    if (d / "samples.npz").exists():
        z = np.load(d / "samples.npz")
        out.update(samples=z["samples"], snap_iters=z["snap_iters"],
                   snap_samples=z["snap_samples"])
    return out


def load_model(target_name: str, method_name: str, seed: int = 0, tag: str = ""):
    """Rebuild a trained method from its cached config and weights."""
    d = run_dir(target_name, method_name, seed, tag)
    cfg = json.loads((d / "metrics.json").read_text())["config"]
    method = build_method(method_name, get_target(target_name), cfg)
    method.load_state_dict(torch.load(d / "model.pt"))
    method.eval()
    return method
