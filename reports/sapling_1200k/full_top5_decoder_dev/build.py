"""Verify the completed full-pass training export, then build its original style-two int8 actor."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from tree_win_gate import digest, sapling_shapes

ACTORS = ROOT / "data/weights/sapling_1200k"
WEIGHTS = ACTORS / "full_top5_decoder.npz"
PARENT = ACTORS / "cash_v3.npz"
PACK = ACTORS / "full_top5_decoder_agent.py"
LOG = HERE.parent / "full_pass_training.json"
PARENT_SHA = "df0cf175db407712f31c188184c1395c0ac9b9c26ddc6b37b9dacda9fc43e5be"


def main():
    report = json.loads(LOG.read_bytes())
    recipe = report["recipe"]
    if (digest(WEIGHTS) != report["weight_sha256"] or digest(PARENT) != PARENT_SHA
            or recipe["parent"] != PARENT_SHA or recipe["train_roots"] != 47168
            or recipe["code"] != digest(ROOT / "code/sapling_full_pass.py")):
        raise ValueError("weight, training pass or frozen parent identity differs")
    with np.load(WEIGHTS, allow_pickle=False) as weights, np.load(PARENT, allow_pickle=False) as parent:
        meta = json.loads(str(weights["meta"]))
        shapes = sapling_shapes(meta)
        if set(weights.files) != (set(shapes) | {"meta"}) or set(weights.files) != set(parent.files):
            raise ValueError("NPZ tensor keys differ")
        if meta["full_pass"] != recipe or meta["training_status"] != "full_top5_decoder_pass":
            raise ValueError("NPZ training metadata differs from completed report")
        changed = []
        count = 0
        for name, shape in shapes.items():
            value = weights[name]
            if (value.shape != shape or value.shape != parent[name].shape
                    or not np.isfinite(value).all()):
                raise ValueError(f"invalid tensor {name}")
            count += value.size
            if not np.array_equal(value, parent[name]):
                if not name.startswith(tuple(recipe["decoder"])):
                    raise ValueError(f"frozen nondecoder changed: {name}")
                changed.append(name)
        if count != 1206476 or not changed:
            raise ValueError("unexpected architecture or zero changed tensors")
    # cnet2_pack writes an extra dequantized NPZ alongside the input. Keep intermediates in scratch.
    with tempfile.TemporaryDirectory(dir=ACTORS) as tmp:
        source = Path(tmp) / "source.npz"
        actor = source.with_suffix(".py")
        shutil.copyfile(WEIGHTS, source)
        details = subprocess.check_output(
            [sys.executable, str(ROOT / "code/cnet2_pack.py"), str(source), str(actor), "2"],
            env=dict(os.environ, CNET2_QMODE="int8"), text=True)
        build = json.loads(details)
        if build["opts"]["teacher"] != 2 or any(v for k, v in build["opts"].items() if k != "teacher"):
            raise ValueError("unexpected inference options")
        pack_sha = digest(actor)
        if PACK.exists() and digest(PACK) != pack_sha:
            raise ValueError("a different actor already exists")
        if not PACK.exists():
            os.link(actor, PACK)
    identity = {"weight_sha256": report["weight_sha256"], "pack_sha256": pack_sha,
                "parent_npz_sha256": PARENT_SHA, "training_report_sha256": digest(LOG),
                "parameter_count": count, "tensor_keys": len(shapes), "changed_tensors": changed,
                "original_packager_sha256": digest(ROOT / "code/cnet2_pack.py"),
                "embedded_source_sha256": build["src_sha256"], "options": build["opts"],
                "weight_format": "int8", "pack_bytes": PACK.stat().st_size}
    out = HERE / "identity.json"
    content = json.dumps(identity, indent=2, allow_nan=False) + "\n"
    if out.exists() and out.read_text() != content:
        raise ValueError("different development identity already exists")
    if not out.exists():
        temp = out.with_suffix(".tmp")
        temp.write_text(content)
        os.replace(temp, out)
    print(pack_sha, flush=True)


if __name__ == "__main__":
    main()
