"""Fit two base-action intercepts for frozen Sapling V4 on naturally occurring choices."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch

from cnet2_model import CNet2
from sapling_bootstrap import sha
from sapling_cash_train import checked_shard
from sapling_tree_train import ELITE_SHA

PARENT_SHA = "2ce3c214683bc4d132ca7502e4864c5fcd7961840e13354671bf7c1173b13a00"
TREE_SHA = "3971262bc0eeca0877b1766eec3524c005ead2a660dfd12edcbe7b3be0211c91"
SEED, MAX_TRAIN_SLOTS = 551192, 4096


@torch.inference_mode()
def score_source(path, split, net, trainer):
    games = [g for g in json.loads(path.read_bytes())["games"] if g["split"] == split]
    indices = []
    offsets = [0]
    for game in games:
        with checked_shard(path, game) as z:
            y, legal = z["y"][:, 10:12], z["legal"][:, 10:12]
            free = (y >= 0) & (legal.sum(-1) >= 2)
            if np.any(free & ~legal[..., :3].all(-1)):
                raise ValueError("expected all three base actions on each free slot")
            idx = np.argwhere(free)
        indices.append(idx)
        offsets.append(offsets[-1] + len(idx))
    population = offsets[-1]
    if not population:
        raise ValueError("no free base actions")
    chosen = np.arange(population)
    if split == "train" and population > MAX_TRAIN_SLOTS:
        chosen = np.sort(np.random.default_rng(SEED).choice(population, MAX_TRAIN_SLOTS, replace=False))
    scores, targets, game_ids = [], [], []
    for game, idx, first, stop in zip(games, indices, offsets, offsets[1:]):
        rows_and_slots = idx[chosen[(chosen >= first) & (chosen < stop)] - first]
        if not len(rows_and_slots):
            continue
        rows = np.unique(rows_and_slots[:, 0])
        with checked_shard(path, game) as z:
            d = {k: z[k][rows] for k in trainer.LOAD}
        if not np.all(d["teacher"] == 2):
            raise ValueError("wrong style slot")
        ref = np.searchsorted(rows, rows_and_slots[:, 0])
        s = rows_and_slots[:, 1] + 10
        parts = []
        for start in range(0, len(rows), 48):
            b = trainer.to_dev(trainer.fetch(d, np.arange(start, min(start + 48, len(rows)))), aug=False)
            parts.append(net(b)[0][:, 10:12, :3].cpu().numpy())
        scores.append(np.concatenate(parts)[ref, s - 10])
        targets.append(d["y"][ref, s])
        game_ids.append(np.full(len(ref), game["seed"], np.int64))
    logits, y, ids = np.concatenate(scores).astype(float), np.concatenate(targets), np.concatenate(game_ids)
    if not np.isfinite(logits).all() or (y < 0).any() or (y > 2).any():
        raise ValueError("nonfinite logits or illegal base actions")
    return dict(logits=logits, y=y, ids=ids, population=population)


def probabilities(sample, delta):
    z = sample["logits"] + delta
    logp = z - logsumexp(z, axis=1, keepdims=True)
    return logp, np.exp(logp)


def fit(train):
    # Fix the NONE logit. Equal weight per source, natural frequency per slot.
    def objective(x):
        delta = np.r_[0., x]
        ce, gradient = 0., np.zeros(3)
        for sample in train.values():
            y = sample["y"]
            logp, p = probabilities(sample, delta)
            ce -= logp[np.arange(len(y)), y].mean() / len(train)
            gradient += (p.mean(axis=0) - np.bincount(y, minlength=3) / len(y)) / len(train)
        return ce, gradient[1:]
    result = minimize(objective, np.zeros(2), jac=True, method="BFGS", options={"gtol": 1e-9})
    if not np.isfinite(result.x).all() or np.abs(result.jac).max() > 1.e-6:
        raise RuntimeError(f"failed intercept fit: {result.message}; gradient={result.jac}")
    return np.r_[0., result.x].astype(np.float32)


def measure(sample, delta):
    y, ids = sample["y"], sample["ids"]
    logp, p = probabilities(sample, delta)
    pred = p.argmax(-1)
    games = np.unique(ids)
    matrix = np.zeros((3, 3), int)
    np.add.at(matrix, (y, pred), 1)
    return dict(population=sample["population"], slots=len(y), games=len(games),
                action_order=["none", "collector", "planter"], teacher_by_predicted=matrix.tolist(),
                agreement=float((pred == y).mean()), mean_game_agreement=float(np.mean([
                    (pred[ids == g] == y[ids == g]).mean() for g in games])),
                ce=float(-logp[np.arange(len(y)), y].mean()),
                brier=float(np.square(p - np.eye(3)[y]).sum(-1).mean()))


def main(a):
    if a.out.exists() or a.receipt.exists():
        raise FileExistsError("fresh output names required")
    if any(k.startswith(("BC_", "CNET2_")) for k in os.environ):
        raise ValueError("unset training environment overrides")
    os.environ["BC_DEVICE"] = "cpu"
    import cnet2_train as trainer
    torch.set_num_threads(1)
    started = time.monotonic()
    raw = a.init.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError("expected frozen V4")
    manifests = {"expert_in_student_games": a.tree, "top5_h2h": a.elite}
    if [sha(p) for p in manifests.values()] != [TREE_SHA, ELITE_SHA]:
        raise ValueError("source identity mismatch")
    seen = set()
    for manifest in manifests.values():
        for g in json.loads(manifest.read_bytes())["games"]:
            if g["seed"] in seen or g["split"] not in ("train", "val"):
                raise ValueError("repeated map or unknown split")
            seen.add(g["seed"])
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        state = {k:z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    net = CNet2(**meta["cfg"])
    net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
    net.eval()
    train = {k:score_source(p, "train", net, trainer) for k,p in manifests.items()}
    delta = fit(train)
    bias = state["bhead.bias"]
    new_bias = bias + delta
    actual = new_bias.astype(float) - bias.astype(float)
    meta["training_status"] = "tree_v4_base_intercept_calibration"
    meta["base_calibration"] = dict(parent_sha256=PARENT_SHA, tree_manifest_sha256=TREE_SHA,
                 elite_manifest_sha256=ELITE_SHA, implementation_sha256=sha(__file__),
                 seed=SEED, max_train_slots_per_source=MAX_TRAIN_SLOTS, train_source_weights=[.5, .5],
                 offset=actual.tolist(), changed_tensor="bhead.bias", fixed_action="none",
                 objective="source-balanced natural per-choice base cross entropy")
    # Seal candidate bytes before loading any validation examples.
    a.out.parent.mkdir(exist_ok=True, parents=True)
    a.receipt.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as td, tempfile.TemporaryDirectory(dir=a.receipt.parent) as rd:
        weight, receipt = Path(td)/"model.npz", Path(rd)/"receipt.json"
        state["bhead.bias"] = new_bias
        np.savez(weight, **state, meta=np.array(json.dumps(meta)))
        with np.load(weight, allow_pickle=False) as check, np.load(io.BytesIO(raw), allow_pickle=False) as orig:
            changed = [k for k in state if not np.array_equal(orig[k], check[k])]
        if changed != ["bhead.bias"]:
            raise ValueError(f"unexpected tensor diff {changed}")
        candidate_hash = sha(weight)
        validation = {k:score_source(p, "val", net, trainer) for k,p in manifests.items()}
        # The parent net retains the original bias because new_bias was allocated.
        if not np.array_equal(net.bhead.bias.detach().numpy(), bias):
            raise RuntimeError("parent network mutated during the calibration audit")
        results = {split: {k:dict(parent=measure(s, np.zeros(3)), candidate=measure(s, actual))
                           for k,s in datasets.items()} for split,datasets in (("train",train), ("val",validation))}
        report = dict(schema="sapling-base-intercepts/1", trained_sha256=candidate_hash, **meta["base_calibration"],
                      changed_tensors=changed, measured=results, elapsed_seconds=time.monotonic()-started)
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        os.link(receipt, a.receipt)
        try:
            os.link(weight, a.out)
        except BaseException:
            a.receipt.unlink()
            raise
    print(report["trained_sha256"], actual.tolist(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("init", "tree", "elite", "out", "receipt"):
        p.add_argument("--"+arg, type=Path, required=True)
    main(p.parse_args())
