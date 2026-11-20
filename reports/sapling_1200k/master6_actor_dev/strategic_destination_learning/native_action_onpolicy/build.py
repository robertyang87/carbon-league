"""Replace only the verified last-layer byte array in the existing one-file Sapling emitter."""
import argparse
import base64
import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "native_action_policy"
PARENT = HERE.parent / "bc.py"
OLD_PACKAGE = "e830229d23817cdcb8ad7e7e392cb3ae57cb1e04503e70ecefe91bf446fc20d4"
PARENT_SHA = "ede48ad9770d357cbe4a4be6c24dafebc910be1e46a95c508c706e009fd713a0"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def assignment(arr):
    raw = base64.b64encode(arr.astype("<f4").tobytes()).decode()
    return f'_NATIVE_W1 = np.frombuffer(base64.b64decode("{raw}"), dtype="<f4").reshape({arr.shape!r})'

def build(output, weights, behavior=False):
    origin = SOURCE / "candidate.py"
    if sha(origin) != OLD_PACKAGE or sha(PARENT) != PARENT_SHA:
        raise ValueError("Updated source policy without registration")
    with np.load(SOURCE / "head.npz",allow_pickle=False) as orig:
        base = {k: orig[k] for k in orig.files}
    if behavior:
        base_zero = dict(base, W1=np.zeros_like(base["W1"]))
        if weights.exists():
            with np.load(weights,allow_pickle=False) as saved:
                if set(saved)!=set(base_zero) or not all(np.array_equal(saved[k],v) for k,v in base_zero.items()):
                    raise ValueError("Different behavior weights already exist")
        else:
            np.savez(weights,**base_zero)
    with np.load(weights,allow_pickle=False) as checkpoint:
        for k in ("W0","B0"):
            if not np.array_equal(base[k],checkpoint[k]):
                raise ValueError("Frozen feature layer changed")
        w=checkpoint["W1"]
    if w.shape!=(24,1) or w.dtype!=np.float32 or not np.isfinite(w).all():
        raise ValueError("Unsupported final-layer coefficients")
    text=origin.read_text()
    old=assignment(base["W1"])
    if text.count(old)!=1:
        raise ValueError("Expected a unique original weight declaration")
    text=text.replace(old,assignment(w))+"\n"+(HERE/"telemetry.py").read_text()
    if len(text.encode())>10_000_000:
        raise ValueError("Tournament single-file limit exceeded")
    if output.exists() and output.read_bytes()!=text.encode():
        raise ValueError("Different release at the output path")
    output.write_text(text)
    return sha(output)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--trained",action="store_true")
    args=parser.parse_args()
    if args.trained:
        print(build(HERE/"trained.py", HERE/"trained.npz"))
    else:
        print(build(HERE/"sampler.py", HERE/"behavior.npz",behavior=True))
