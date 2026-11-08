"""Registry of VI methods."""

from .base import VIMethod
from .adversarial import AdversarialVI
from .gaussian_vi import GaussianVI
from .kivi import KernelImplicitVI
from .sivi import SemiImplicitVI

METHODS = {cls.name: cls for cls in (GaussianVI, AdversarialVI, KernelImplicitVI, SemiImplicitVI)}


def build_method(name: str, target, cfg: dict) -> VIMethod:
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; choose from {list(METHODS)}")
    return METHODS[name](target, cfg)
