"""Independent Sapling initialization, no source weights.

  python code/init_sapling.py --seed 551187 --out /tmp/sapling.npz
  python code/init_sapling.py --preset sapling-1.2m --seed 551187 --out /tmp/sapling_large.npz

The unchanged default has 551,187 parameters. The 1.2m target-budget preset has
1,206,476 parameters: C=93, NB=6, G=3, with the same D=96/L=1 decoder and attention.
G controls convolutional GroupNorm only; 93 channels form 3 groups of 31.
The preset name is a target tier, not an exact parameter count.

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
DEFAULT_PRESET = "sapling-551k"
SAPLING_PRESETS = {
    DEFAULT_PRESET: dict(cfg=SAPLING_CONFIG, params=EXPECTED_PARAMS),
    "sapling-1.2m": dict(cfg=dict(SAPLING_CONFIG, C=93, NB=6, G=3), params=1_206_476),
}


def resolve_preset_metadata(meta):
    """Identify only the two registered shapes; refuse misleading metadata.

    Old Sapling files need no `preset` field. Missing TA means zero, as in the
    original CNet2 contract; optional ZI=0/TNH=4 are default-equivalent only.
    Tensor keys/shapes and the actual parameter count must still be checked by
    the caller when loading the state dictionary.
    """
    if not isinstance(meta, dict) or not isinstance(meta.get("cfg"), dict):
        raise ValueError("Sapling weights require an object meta.cfg")
    cfg = meta["cfg"]
    if set(cfg) - (set(SAPLING_CONFIG) | {"ZI", "TNH"}):
        raise ValueError("unknown fields in Sapling cfg")
    if any(type(value) is not int for value in cfg.values()):
        raise ValueError("Sapling cfg values must be integers, not booleans")
    if cfg.get("ZI", 0) != 0 or cfg.get("TNH", 4) != 4:
        raise ValueError("non-default ZI/TNH is not a registered Sapling preset")
    for name, spec in SAPLING_PRESETS.items():
        if all(cfg.get(k, 0 if k == "TA" else None) == v for k, v in spec["cfg"].items()):
            if "preset" in meta and meta["preset"] != name:
                raise ValueError("Sapling preset name does not match cfg")
            if "params" in meta and (type(meta["params"]) is not int or meta["params"] != spec["params"]):
                raise ValueError("Sapling parameter-count metadata does not match cfg")
            return name, dict(spec["cfg"]), spec["params"]
    raise ValueError("unsupported Sapling cfg; expected sapling-551k or sapling-1.2m")


def initialize(output, seed, preset=DEFAULT_PRESET):
    output = Path(output).expanduser().absolute()
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if output.suffix != ".npz":
        raise ValueError("output must have the .npz suffix")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if preset not in SAPLING_PRESETS:
        raise ValueError(f"unknown Sapling preset: {preset!r}")
    spec = SAPLING_PRESETS[preset]

    import torch
    from cnet2_model import CNet2
    from rl2_ppo_cnet2 import Cnet2Adapter
    from rl2_ppo_adapters import save_npz

    # CPU-only construction; preserve a caller's RNG and avoid accelerator RNGs.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CNet2(**spec["cfg"]).cpu().eval()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != spec["params"]:
        raise RuntimeError(f"Sapling contract changed: expected {spec['params']} "
                           f"parameters, got {count}")
    metadata = dict(model_name="Sapling", seed=seed, params=count, untrained=True, preset=preset)
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
    parser.add_argument("--preset", choices=tuple(SAPLING_PRESETS), default=DEFAULT_PRESET,
                        help="architecture tier; default preserves the original 551187-parameter model")
    args = parser.parse_args(argv)
    try:
        receipt = initialize(args.out, args.seed, args.preset)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
