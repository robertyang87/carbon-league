"""Validate all natural-frequency masks and independently rescore shipped probabilities in PyTorch."""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
os.environ["BC_DEVICE"] = "cpu"
import cnet2_train as trainer
from cnet2_model import CNet2


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_teacher():
    path = ROOT / "code/packs/ppou2_m_it50.py"
    if sha(path) != "57e2557665636a1bbc92ae46a4c33ee9dac609efcffefa1c30df1233746c2211":
        raise ValueError("teacher bytes changed")
    spec = importlib.util.spec_from_file_location("teacher_source", path)
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    w = p._dequantize(zlib.decompress(base64.b64decode(p._B1)), p._Q1)
    net = CNet2(**json.loads(p._M1)["cfg"])
    missing, unexpected = net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in w.items()}, strict=False)
    missing_heads = {h+"."+k for h in ("utility", "margin", "wdl") for k in ("weight","bias")}
    if unexpected or not set(missing) <= missing_heads:
        raise ValueError((missing, unexpected))
    return net.eval()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    m = json.loads(args.manifest.read_text())
    if [g["seed"] for g in m["games"]] != list(range(620176000,620176024)):
        raise ValueError("map identities differ from frozen split")
    net = load_teacher()
    max_prob = 0.
    examples = 0
    checked_slots = 0
    checked_conflict_tail = 0
    max_sum_error = 0.
    counts = defaultdict(Counter)
    for i,g in enumerate(m["games"]):
        source = args.manifest.parent/g["shard"]
        if sha(source) != g["sha256"] or g["split"] != ("train" if i<18 else "val"):
            raise ValueError("changed shard or map split")
        with np.load(source, allow_pickle=False) as z:
            y, prob, valid, best, msk = (z[k] for k in ("y","teacher_probs","legal","teacher_argmax","soft_mask"))
            active = z["tok_mask"][:,:12]
            steps = z["step"]
            lower, upper = int(steps[-1]) + 1, min(int(steps[-1]) + 3, 299)
            if "transitions_range" in g and g["transitions_range"] != [lower, upper]:
                raise ValueError("declared terminal-time bounds differ from observed samples")
            if g["transitions"] is None:
                if "transitions_range" not in g or lower == upper or not m.get("recovery"):
                    raise ValueError("missing terminal time has no recovery evidence")
                stop = lower
            else:
                stop = g["transitions"]
                if not lower <= stop <= upper:
                    raise ValueError("terminal time conflicts with recorded last sample")
            if not np.array_equal(steps, np.arange(i%3, stop,3)):
                raise ValueError("sampling mask differs from fixed, event-independent plan")
            if not np.array_equal(y>=0,active) or not np.array_equal(best>=0,active):
                raise ValueError("inactive token labeling mismatch")
            if not np.array_equal(msk, (valid.sum(-1)>=2) & active):
                raise ValueError("a naturally available decision was excluded or duplicated")
            if np.any(prob[~valid]) or not np.isfinite(prob).all() or np.any(prob<0):
                raise ValueError("invalid support")
            if not np.array_equal(prob.argmax(-1)[active],best[active]):
                raise ValueError("distribution/argmax mismatch")
            if not np.all(np.take_along_axis(valid,y.clip(min=0)[...,None],axis=2).squeeze(-1)[active]):
                raise ValueError("executed a masked decision")
            max_sum_error = max(max_sum_error, float(np.abs(prob.sum(-1)-active.astype(float)).max()))
            if max_sum_error > 1.e-6:
                raise ValueError("distribution mass mismatch")
            conflict = active & (best != y)
            if g["behavior"] == "it50 bare style 2" and conflict.any():
                raise ValueError("it50 greedy trajectory differs from its own teacher")
            downstream = (np.cumsum(conflict,axis=1) - conflict.astype(int) > 0) & msk
            # Prespecified three temporal positions, plus earliest available root with
            # a later slot whose conditioning action differed from greedy it50.
            indices = sorted({0, len(y)//2, len(y)-1} |
                             (set(np.flatnonzero(downstream.any(1))[:1].tolist())))
            data = {k:z[k][indices] for k in trainer.FIELDS}
            q = net(trainer.to_dev(data, False))[0].softmax(-1).numpy()
            delta = np.abs(q-prob[indices])
            error = float(delta[active[indices]].max())
            if error > 2.e-5:
                raise ValueError(f"independent checkpoint soft distribution mismatch: {g['seed']} {error}")
            max_prob = max(max_prob,error)
            checked_conflict_tail += int(downstream[indices].sum())
            examples += len(indices)
            checked_slots += int(active[indices].sum())
            peak = prob.max(-1)[msk]
            q_free = prob[msk].astype(np.float64)
            entropy_sum = float(-(q_free * np.log(q_free.clip(min=np.finfo(float).tiny))).sum())
            counts[g["split"]].update(entropy_sum_nats=entropy_sum, teacher_max_probability_sum=float(peak.sum(dtype=np.float64)),
                                      teacher_max_probability_below_0_9=int((peak < 0.9).sum()),
                                      games=1, roots=len(y), free_slots=int(msk.sum()),
                                      subsequent_free_slots_after_non_greedy_execution=int(downstream.sum()),
                                      executed_differs=int(conflict.sum()))
    if not checked_conflict_tail:
        raise ValueError("no conditional downstream of an off-teacher choice checked")
    for group in counts.values():
        group["mean_entropy_nats_per_free_slot"] = group.pop("entropy_sum_nats") / group["free_slots"]
        group["mean_teacher_max_probability"] = group.pop("teacher_max_probability_sum") / group["free_slots"]
    with args.out.open("x") as f:
        json.dump(dict(manifest_sha256=sha(args.manifest), groups=counts, torch_checked_roots=examples,
                       torch_checked_slots=checked_slots, checked_subsequent_free_slots_after_non_greedy=checked_conflict_tail,
                       max_abs_probability_error=max_prob, maximum_normalization_error=max_sum_error,
                       meaning="Full mask and timing checks. PyTorch rescores the original dequantized it50 weights conditioned on the stored real action prefix on each selected root."),
                  f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
