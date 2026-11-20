"""Materialize signed model inputs and use the established map/seat exposure ledger."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / ".git").exists())
source = ROOT / "reports/sapling_1200k/master6_actor_dev/strategic_destination_learning/development/workflow_cli.py"
spec = importlib.util.spec_from_file_location("shared_sparse_checkout", source)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.HERE = HERE
shared.SUITE = "master3s2-cold-repair-development.yml"
shared.EXPOSURE_STEP = "Play one registered V2 game"
shared.ARMS = ("cold_repair",)


def prior():
    guard = shared.prior()
    guard.prefix = lambda seed, seat: f"m3s2-cold-{seed}-{seat}"
    return guard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "guard", "exposure"))
    parser.add_argument("--digest")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seat", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.mode == "prepare":
        shared.prepare(args.digest)
    elif args.mode == "guard":
        prior().guard(args.seed, args.seat)
    else:
        prior().exposure(args.out, args.seed, args.seat)
