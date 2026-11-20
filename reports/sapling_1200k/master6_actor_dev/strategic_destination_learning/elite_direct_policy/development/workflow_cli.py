"""Use the established sparse input and past-exposure checks with the replication names."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("strategic_sparse",
                                           HERE.parents[1]/"development/workflow_cli.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
prior.HERE = HERE
prior.SUITE = "sapling-elite-bc-comparison.yml"
prior.EXPOSURE_STEP = "Play elite and control against original"
prior.ARMS = ("control", "elite")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "guard", "exposure"))
    parser.add_argument("--digest")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seat", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.mode == "prepare":
        prior.prepare(args.digest)
    else:
        checker = prior.prior()
        checker.prefix = lambda seed, seat: f"elite-dev-{seed}-{seat}"
        if args.mode == "guard":
            checker.guard(args.seed, args.seat)
        else:
            checker.exposure(args.out, args.seed, args.seat)
