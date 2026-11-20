"""Materialize signed model inputs and use the established map/seat exposure ledger."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent / "development/workflow_cli.py"
spec = importlib.util.spec_from_file_location("shared_sparse_checkout", source)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.HERE = HERE
shared.SUITE = "sapling-native-action-onpolicy-intake.yml"
shared.EXPOSURE_STEP = "Collect one registered on-policy training trajectory"
shared.ARMS = ("collector",)


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
        shared.prior().guard(args.seed, args.seat)
    else:
        shared.prior().exposure(args.out, args.seed, args.seat)
