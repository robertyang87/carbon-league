"""Independent Sapling initialization: 551,187 parameters, no source weights.

  python code/init_sapling.py --seed 551187 --out /tmp/sapling.npz

Uses the existing CNet2 NPZ/NumPy/training contract. All heads are untrained,
including utility/WDL/margin. CPU initialization is reproducible with the same
seed and software versions. Existing outputs are never overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

SAPLING_CONFIG = dict(C=64, NB=4, TA=0, D=96, NH=4, L=1, E=32, G=8)
EXPECTED_PARAMS = 551_187


def initialize(output, seed):
    output = Path(output).expanduser().absolute()
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if output.suffix != ".npz":
        raise ValueError("output must have the .npz suffix")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    import torch
    from cnet2_model import CNet2
    from rl2_ppo_cnet2 import Cnet2Adapter
    from rl2_ppo_adapters import save_npz

    # CPU-only construction; preserve a caller's RNG and avoid accelerator RNGs.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CNet2(**SAPLING_CONFIG).cpu().eval()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMS:
        raise RuntimeError(f"Sapling contract changed: expected {EXPECTED_PARAMS} "
                           f"parameters, got {count}")
    metadata = dict(model_name="Sapling", seed=seed, params=count, untrained=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Export completely on the destination filesystem. link() atomically
    # refuses an existing path, including a concurrent writer or symlink.
    with tempfile.TemporaryDirectory(prefix=".sapling-", dir=output.parent) as stage:
        staged = Path(stage) / "weights.npz"
        save_npz(Cnet2Adapter().export(model), str(staged), metadata)
        os.link(staged, output)
    return dict(metadata, cfg=dict(model.cfg), output=str(output),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True, help="new .npz path; never overwritten")
    args = parser.parse_args(argv)
    try:
        receipt = initialize(args.out, args.seed)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
