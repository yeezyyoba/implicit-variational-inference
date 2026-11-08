"""Small shared helpers: seeding, config loading, paths."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: Path | str | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def method_config(method: str, overrides: dict | None = None, path=None) -> dict:
    """Merge the `common` block with a method's block, then apply overrides.

    Overrides are flat ``{"key": value}`` pairs (already scoped to the method).
    """
    cfg = load_config(path)
    merged = copy.deepcopy(cfg.get("common", {}))
    merged.update(cfg.get(method, {}))
    merged.update(overrides or {})
    return merged


def parse_overrides(pairs: list[str] | None, method: str) -> dict:
    """Parse CLI overrides like ``disc_steps=1`` or ``adversarial.disc_steps=1``.

    Overrides scoped to a different method are ignored for this method.
    """
    out = {}
    for pair in pairs or []:
        key, value = pair.split("=", 1)
        if "." in key:
            scope, key = key.split(".", 1)
            if scope != method:
                continue
        out[key] = yaml.safe_load(value)
    return out
