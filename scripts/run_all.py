"""Train every requested (target, method, seed) combination, skipping cached runs.

Examples
--------
Everything (the full comparison):
    python scripts/run_all.py
One run:
    python scripts/run_all.py --targets banana --methods gaussian_vi --seeds 0
An ablation, stored under its own tag:
    python scripts/run_all.py --targets banana --methods adversarial \
        --set adversarial.disc_steps=1 --tag k1
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivi.experiment import run_one  # noqa: E402
from ivi.methods import METHODS  # noqa: E402
from ivi.targets import TARGETS  # noqa: E402
from ivi.utils import parse_overrides  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tag", default="", help="name for a non-default configuration")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. adversarial.disc_steps=1 or n_iters=2000")
    ap.add_argument("--force", action="store_true", help="re-run even if cached")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--time-budget", type=float, default=None, metavar="SECONDS",
                    help="pause (checkpoint) when this much wall time has passed; "
                         "re-run the same command to resume")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    deadline = time.time() + args.time_budget if args.time_budget else None
    for target in args.targets:
        for method in args.methods:
            for seed in args.seeds:
                out = run_one(target, method, seed, tag=args.tag,
                              overrides=parse_overrides(args.set, method), force=args.force,
                              deadline=deadline)
                if out is None:  # paused by the time budget
                    sys.exit(2)
    print("ALL_DONE")


if __name__ == "__main__":
    main()
