"""Sanity checks for the targets and their ground truth. Run: python -m pytest tests/
(or simply: python tests/test_targets.py)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivi.targets import TARGETS, get_target  # noqa: E402


def test_grid_normalises_to_one():
    for name in TARGETS:
        gt = get_target(name).ground_truth()
        mass = np.exp(gt["log_density"]).sum() * gt["dx"] * gt["dy"]
        assert abs(mass - 1.0) < 1e-6, (name, mass)


def test_grid_resolution_converged():
    # log Z should barely move when the grid is refined -> the box holds the mass
    for name in TARGETS:
        t = get_target(name)
        assert abs(t.ground_truth(200)["log_Z"] - t.ground_truth(400)["log_Z"]) < 1e-3, name


def test_truth_samples_match_moments():
    # Banana: z1 ~ N(0, a^2) exactly, and E[z2] = 0 by construction
    t = get_target("banana")
    s = t.sample_truth(200_000, seed=1)
    assert abs(s[:, 0].std() - t.a) < 0.02
    assert abs(s[:, 1].mean()) < 0.02


def test_prior_plus_likelihood_is_target():
    z = torch.randn(10, 2)
    for name in TARGETS:
        t = get_target(name)
        assert torch.allclose(t.log_prior(z) + t.log_lik(z), t.log_unnorm(z), atol=1e-5)


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
