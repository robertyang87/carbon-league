"""Run a frozen four-iteration win-only PPO pilot from a published Sapling parent."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
POOL = [
    ["policy_v0_5.py", 1.0],
    ["policy_raid_v0_1.py", 0.5],
    ["../data/weights/sapling_1200k/cash_v3_agent.py", 1.0],
    ["packs/ppou2_m_it50.py", 0.5],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh checkpoint directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--curriculum", choices=("tree", "cash-selfplay"), default="tree",
                        help="cash-selfplay starts at V3, explores less and includes mirrored self-play")
    args = parser.parse_args()
    selfplay = args.curriculum == "cash-selfplay"
    parent = ROOT / "data/weights/sapling_1200k" / ("cash_v3.npz" if selfplay else "tree_v4.npz")
    receipt_path = ROOT / "reports/sapling_1200k" / ("cash_v3_training.json" if selfplay else "tree_v4_training.json")
    if not parent.is_file() or not receipt_path.is_file():
        parser.error("the published parent weight and training receipt are required")
    receipt = json.loads(receipt_path.read_text())
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    if parent_sha != receipt["trained_sha256"]:
        parser.error("parent differs from the published training receipt")
    if any(key.startswith(("CNET2_", "RL2_")) for key in os.environ):
        parser.error("clear CNET2_* and RL2_* overrides for this fixed pilot")
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", PYTHONHASHSEED="0", LEARNER_THREADS="2", CNET2_TEACHER="2")
    cmd = [sys.executable, "-u", "rl2_ppo_train.py", "--adapter", "cnet2", "--init", str(parent),
           "--out", str(args.out.resolve()), "--device", args.device, "--games", "8", "--workers", "2",
           "--lockstep", "2", "--overlap", "0", "--iters", "4", "--vwarm", "1", "--sep-value", "1",
           "--reward", "win", "--gamma", "1", "--lam", "1", "--rscale", "1", "--win-bonus", "0",
           "--value-head", "tanh", "--temp", "0.30" if selfplay else "0.85", "--epochs", "1", "--ratio", "joint",
           "--lr", "0.00003", "--vlr", "0.0001", "--kl", "0.03", "--ent", "0.001",
           "--dec-frac", "0.25", "--pre-mb", "24", "--mb", "24", "--eval-every", "0",
           "--ckpt-every", "4", "--snap-every", "9999", "--seed0", "620165000" if selfplay else "620164000",
           "--sampler-seed", "551193" if selfplay else "551192", "--self-prior", "3.0" if selfplay else "0.0",
           "--pool-authoritative", "--pool", json.dumps(POOL)]
    with (args.out / "pilot_identity.json").open("x") as f:
        json.dump({"curriculum": args.curriculum, "parent_sha256": parent_sha, "command": cmd, "pool_sha256": {
            entry: hashlib.sha256((ROOT / "code" / entry).read_bytes()).hexdigest() for entry, _ in POOL}},
            f, indent=2)
        f.write("\n")
    subprocess.run(cmd, cwd=ROOT / "code", env=env, check=True)


if __name__ == "__main__":
    main()
