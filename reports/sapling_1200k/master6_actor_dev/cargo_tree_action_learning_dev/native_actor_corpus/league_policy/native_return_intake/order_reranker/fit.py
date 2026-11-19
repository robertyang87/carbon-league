"""Cross-fit a bounded scorer on six disjoint terrains with complete official returns."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import collections
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from features import _order_rows, _order_logit_delta
from rerank import export_pack, SOURCE_SHA

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent
ROOT = next(p for p in HERE.parents if (p / ".git").exists())
PREREG = "89831b8160209491a434ba72006a513d5616d935"
PLAN_SHA = "9bb0f2ce651061037bdf0ecae35bb6a0351bfbbf25809d6280eec5a308c8e291"
MANIFEST_SHA = "63daa51e127aafea36401ebdce6400de60faa772d3929a1803ce768e121be5b4"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(file, data):
    dest = HERE / file
    raw = (json.dumps(data, indent=2, allow_nan=False, sort_keys=True) + "\n").encode()
    if dest.exists() and dest.read_bytes() != raw:
        raise ValueError("Conflicting training artifact " + file)
    stage = dest.with_suffix(dest.suffix + ".tmp")
    stage.write_bytes(raw)
    stage.replace(dest)


def source_data():
    path = HERE / "protocol.json"
    registered = subprocess.check_output(["git", "-C", str(ROOT), "show", PREREG + ":" + str(path.relative_to(ROOT))])
    manifest = (INTAKE / "hosted/training_manifest.json").read_bytes()
    if registered != path.read_bytes() or sha(registered) != PLAN_SHA or sha(manifest) != MANIFEST_SHA:
        raise ValueError("Training source changed")
    m = json.loads(manifest)
    if not m["training_eligible"] or m["policy_sha256"] != SOURCE_SHA:
        raise ValueError("Wrong source behavior")
    episodes = m["episodes"]
    if len(episodes) != 24 or len({e["terrain"] for e in episodes}) != 6:
        raise ValueError("Incomplete map group")
    examples = []
    for ep in episodes:
        for event in ep["events"]:
            if not event["supported"]:
                if len(event["options"]) != 1 or event["probabilities"] != [1.0]:
                    raise ValueError("Malformed singleton support")
                continue
            obs = event["public_observation"]
            if "full_carbon" in obs or int(obs["player"]) != ep["seat"] or obs["step"] != event["step"]:
                raise ValueError("Invalid pre-action features")
            if sum(o == event["issued"] for o in event["options"]) != 1 or event["options"][event["choice"]] != event["issued"]:
                raise ValueError("No unique recorded physical action")
            gap = event["native_logits"][0] - event["native_logits"][1]
            p = 1 / (1 + math.exp(-gap))
            if abs(p - event["probabilities"][0]) > 1e-12:
                raise ValueError("Behavior probability mismatch")
            x = _order_rows(obs, event["uid"], event["options"], event["native_logits"])
            if not np.isfinite(x).all() or x.shape != (2, 32):
                raise ValueError("Invalid public-state features")
            examples.append(dict(seed=ep["seed"], terrain=ep["terrain"], seat=ep["seat"], arm=ep["arm"],
                                 step=event["step"], x=x, gap=gap, old=p,
                                 side=1 if event["choice"] == 0 else -1, advantage=ep["advantage"]))
    if len(examples) != 275:
        raise ValueError("Source support census changed")
    return episodes, examples


def train(episodes, examples, fit_seeds):
    selected = [x for x in examples if x["seed"] in fit_seeds]
    trajectories = [e for e in episodes if e["seed"] in fit_seeds]
    if len(trajectories) != 4 * len(fit_seeds):
        raise ValueError("Incomplete fit units")
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64)
    x = tensor(np.stack([x["x"] for x in selected]))
    side = tensor([x["side"] for x in selected])
    gap = tensor([x["gap"] for x in selected])
    advantages = tensor([x["advantage"] for x in selected])
    dimensions = x.shape[-1]
    rng = np.random.default_rng(7091)
    initial = {"hidden_w": tensor(rng.normal(0, 1 / math.sqrt(dimensions), (dimensions, 8))),
               "hidden_b": torch.zeros(8, dtype=torch.float64), "readout": torch.zeros(8, dtype=torch.float64)}
    param = {k: torch.nn.Parameter(v.clone()) for k, v in initial.items()}
    optimizer = torch.optim.Adam(list(param.values()), lr=0.02, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    denom = 12 * len(trajectories) * float(np.mean([abs(e["advantage"]) for e in trajectories]))
    if denom == 0:
        return {k: v.tolist() for k, v in initial.items()}
    old = torch.nn.functional.logsigmoid(side * gap)
    for _ in range(32):
        optimizer.zero_grad(set_to_none=True)
        hidden = (x @ param["hidden_w"] + param["hidden_b"]).tanh()
        delta = 0.25 * ((hidden[:, 0] - hidden[:, 1]) @ param["readout"]).tanh()
        logp = torch.nn.functional.logsigmoid(side * (gap + delta))
        objective = torch.sum(advantages * (logp - old)) / denom
        penalty = 0.001 * sum((v - initial[k]).square().mean() for k, v in param.items())
        loss = -objective + penalty
        if not torch.isfinite(loss):
            raise ValueError("Invalid fit")
        loss.backward()
        optimizer.step()
    return {k: v.detach().tolist() for k, v in param.items()}


def data_outputs():
    episodes, examples = source_data()
    seeds = sorted({e["seed"] for e in episodes})
    groups = {seed: {e["terrain"] for e in episodes if e["seed"] == seed} for seed in seeds}
    if set(map(len, groups.values())) != {1} or len(set.union(*groups.values())) != 6:
        raise ValueError("Physical maps overlap")
    weights = dict(schema="sapling-bounded-order-weights/1", prereg_sha256=PLAN_SHA,
                   source_manifest_sha256=MANIFEST_SHA, source_package_sha256=SOURCE_SHA,
                   folds={}, fit_maps={}, full={})
    predictions = []
    package_hashes = {}
    area = collections.defaultdict(dict)
    with tempfile.TemporaryDirectory() as temp:
        for seed in seeds:
            fit_seeds = set(seeds) - {seed}
            vals = train(episodes, examples, fit_seeds)
            weights["folds"][str(seed)] = vals
            weights["fit_maps"][str(seed)] = sorted(fit_seeds)
            digest = export_pack(vals, Path(temp) / (str(seed) + ".py"))
            package_hashes[str(seed)] = digest
            arrays = {k: np.asarray(v, dtype=float) for k, v in vals.items()}
            for e in (x for x in examples if x["seed"] == seed):
                delta = _order_logit_delta(e["x"], arrays)
                q0 = 1 / (1 + math.exp(-e["gap"] - delta))
                if not 0 < q0 < 1 or abs(delta) > 0.25:
                    raise ValueError("Support bound failed")
                area[seed, e["seat"], e["arm"]][str(e["step"])] = q0
        for k, q0 in sorted(area.items()):
            predictions.append(dict(seed=k[0], seat=k[1], arm=k[2], q0=q0))
        weights["full"] = train(episodes, examples, set(seeds))
        full_sha = export_pack(weights["full"], HERE / "research_candidate.py")
    save("weights.json", weights)
    save("fold_predictions.json", dict(schema="sapling-held-map-conditional-probs/1", episodes=predictions,
         parent_package_sha256=SOURCE_SHA, held_fold_package_sha256=package_hashes, full_package_sha256=full_sha))
    return full_sha


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    print(data_outputs())
