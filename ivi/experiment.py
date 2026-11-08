"""Training loop, evaluation and per-run caching.

A *run* is one (target, method, seed, tag). Its outputs are written to
    results/runs/<target>/<method>[__<tag>]/seed<k>/
        metrics.json   final metrics + config + timings      (committed)
        history.json   training curves                       (committed)
        samples.npz    final samples + training snapshots    (git-ignored)
        model.pt       trained parameters                    (git-ignored)
        checkpoint.pt  mid-training state, only while paused (git-ignored)
Runs are skipped if metrics.json already exists, and a run paused by a time
budget resumes from its checkpoint, so long sweeps can be done in pieces.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .methods import build_method
from .metrics import grid_divergences, kl_knn, mmd2, mode_coverage
from .targets import get_target
from .utils import RESULTS_DIR, method_config, set_seed

RUNS_DIR = RESULTS_DIR / "runs"


def run_dir(target: str, method: str, seed: int, tag: str = "") -> Path:
    folder = method + (f"__{tag}" if tag else "")
    return RUNS_DIR / target / folder / f"seed{seed}"


def _optimizers(method) -> dict:
    return {k: v for k, v in vars(method).items() if isinstance(v, torch.optim.Optimizer)}


def _save_checkpoint(path: Path, method, it: int, hist, snapshots, step_time: float) -> None:
    flags = {k: v for k, v in vars(method).items()
             if k.startswith("_") and isinstance(v, (bool, int, float)) and k != "_is_full_backward_hook"}
    torch.save(dict(it=it, model=method.state_dict(),
                    optim={k: o.state_dict() for k, o in _optimizers(method).items()},
                    flags=flags, hist=dict(hist), snapshots=snapshots, step_time=step_time,
                    torch_rng=torch.get_rng_state(), np_rng=np.random.get_state()), path)


def _load_checkpoint(path: Path, method):
    ck = torch.load(path, weights_only=False)
    method.load_state_dict(ck["model"])
    for k, o in _optimizers(method).items():
        o.load_state_dict(ck["optim"][k])
    for k, v in ck["flags"].items():
        setattr(method, k, v)
    torch.set_rng_state(ck["torch_rng"])
    np.random.set_state(ck["np_rng"])
    return ck["it"], defaultdict(list, ck["hist"]), ck["snapshots"], ck["step_time"]


def train(method, n_iters: int, log_every: int = 20, eval_every: int = 200,
          n_eval: int = 20000, snapshot_every: int | None = None,
          n_snapshot: int = 3000, verbose: bool = False,
          checkpoint: Path | None = None, deadline: float | None = None):
    """Run optimisation steps up to ``n_iters``; return (history, snapshots, done).

    If ``checkpoint`` exists, training resumes from it. If ``deadline`` (a
    time.time() value) passes, the full state is saved to ``checkpoint`` and the
    function returns done=False; calling it again continues exactly.
    """
    target = method.target
    hist, snapshots, step_time, start = defaultdict(list), [], 0.0, 1
    if checkpoint is not None and checkpoint.exists() and checkpoint.stat().st_size > 0:
        last, hist, snapshots, step_time = _load_checkpoint(checkpoint, method)
        start = last + 1
        if verbose:
            print(f"  resumed from iteration {last}", flush=True)
    for it in range(start, n_iters + 1):
        t0 = time.perf_counter()
        logs = method.step()
        step_time += time.perf_counter() - t0

        if it == 1 or it % log_every == 0:
            hist["iter"].append(it)
            for k, v in logs.items():
                hist[k].append(float(v))
        if it % eval_every == 0 or it == n_iters:
            s = method.sample(n_eval).numpy()
            g = grid_divergences(s, target)
            kl = kl_knn(s, target)["kl_knn"]
            hist["eval_iter"].append(it)
            hist["kl_true"].append(kl)      # binning-free reverse KL to the truth
            hist["kl_grid"].append(g["kl_qp"])
            hist["tv_grid"].append(g["tv"])
            if verbose:
                print(f"  it {it:5d}  loss {logs['loss']:+.3f}  KL_est {logs['kl_est']:.3f}"
                      f"  KL_true {kl:.3f}  TV {g['tv']:.3f}", flush=True)
        if snapshot_every and (it == 1 or it % snapshot_every == 0):
            snapshots.append((it, method.sample(n_snapshot).numpy().astype(np.float32)))
        if deadline is not None and time.time() > deadline and it < n_iters:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            _save_checkpoint(checkpoint, method, it, hist, snapshots, step_time)
            return dict(hist), snapshots, False
    hist["train_time_s"] = step_time
    return dict(hist), snapshots, True


def evaluate(method, n_samples: int = 100000, seed: int = 0) -> tuple[dict, np.ndarray]:
    """Final metrics from samples, plus the method's own KL estimate."""
    target = method.target
    samples = method.sample(n_samples).numpy()
    truth = target.sample_truth(20000, seed=12345)
    out = dict(grid_divergences(samples, target))
    out.update(kl_knn(samples, target))
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
            verbose: bool = True, deadline: float | None = None) -> dict | None:
    """Train + evaluate + save one run. Returns its metrics, or None if it was
    paused at ``deadline`` (re-run the same call to resume)."""
    out_dir = run_dir(target_name, method_name, seed, tag)
    if (out_dir / "metrics.json").exists() and not force:
        if verbose:
            print(f"[skip] {out_dir.relative_to(RESULTS_DIR.parent)} exists")
        return json.loads((out_dir / "metrics.json").read_text())

    cfg = method_config(method_name, overrides, target=target_name)
    set_seed(seed)
    target = get_target(target_name)
    method = build_method(method_name, target, cfg)
    if verbose:
        print(f"[run] target={target_name} method={method_name} seed={seed} tag={tag or '-'}",
              flush=True)

    ckpt = out_dir / "checkpoint.pt"
    if force and ckpt.exists():
        ckpt.write_bytes(b"")  # an empty checkpoint is ignored
    hist, snaps, done = train(
        method, cfg["n_iters"], log_every=cfg["log_every"], eval_every=cfg["eval_every"],
        n_eval=cfg["n_eval_samples"], snapshot_every=cfg.get("snapshot_every"),
        n_snapshot=cfg.get("n_snapshot_samples", 3000), verbose=verbose,
        checkpoint=ckpt, deadline=deadline)
    if not done:
        if verbose:
            print(f"[paused] checkpoint saved; re-run the same command to resume", flush=True)
        return None
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
    if ckpt.exists():
        ckpt.write_bytes(b"")  # emptied rather than deleted (deletion may be disabled)
        try:
            ckpt.unlink()
        except OSError:
            pass
    if verbose:
        print(f"[done] KL(q||p)={metrics['kl_knn']:.3f}  KL(p||q)={metrics['kl_pq']:.3f}  "
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
