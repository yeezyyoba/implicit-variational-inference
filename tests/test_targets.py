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
    # the box-normalised grid should barely move when refined
    for name in TARGETS:
        t = get_target(name)
        assert abs(t.ground_truth(200)["log_Z_box"] - t.ground_truth(400)["log_Z_box"]) < 1e-3


def test_log_Z_matches_closed_form():
    import math
    t = get_target("banana")      # 2*pi*a*s
    assert abs(t.log_Z - math.log(2 * math.pi * t.a * t.s)) < 1e-4
    t = get_target("x_shape")     # 2 components, each 2*pi*sqrt(det)
    assert abs(t.log_Z - math.log(4 * math.pi * math.sqrt(t.var ** 2 - t.cov ** 2))) < 1e-4
    t = get_target("gmm")         # 4 components, each 2*pi*s^2
    assert abs(t.log_Z - math.log(4 * 2 * math.pi * t.mode_std ** 2)) < 1e-4
    for name in TARGETS:          # the plotting box holds (almost) all the mass
        assert get_target(name).ground_truth()["in_box_mass"] > 0.995


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
