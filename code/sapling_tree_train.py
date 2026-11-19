"""Continue the released Sapling V3 actor on economy and tactical decisions.

Three disjoint sources: V3-visited states relabeled by an expert, V3 cash
rehearsal, and actual top-five head-to-head games. Only historical training
weights use the next public observation. All network inputs remain causal.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from cnet2_feat import DEST
from cnet2_model import CNet2
from init_sapling import resolve_preset_metadata
from sapling_bootstrap import sha
from sapling_cash_train import checked_shard, free_slots, load_corrections, metrics
from sapling_elite_bc import load_games

ROOT = Path(__file__).resolve().parents[1]
WEIGHT = ROOT / "data/weights/sapling_1200k/cash_v3.npz"
PARENT_SHA = "df0cf175db407712f31c188184c1395c0ac9b9c26ddc6b37b9dacda9fc43e5be"
CASH_SHA = "ec0827c112e711a63ad75e8f764f5d10d988d2b7d0bc17979b8e4bb3414a6f40"
ELITE_SHA = "566bd7437a997078a751de6a7f0c9d0f8af1f41a901d6c0ea66b9f6192b16a9b"
SEED, STEPS, LR = 551191, 320, 0.00018
TREE_MASKS = ("planter", "recruit", "return", "defense")
# (data source, label subset, roots per update, fraction of total loss)
GROUPS = (("tree", "planter", 4, .13), ("tree", "recruit", 2, .07),
          ("tree", "return", 3, .10), ("tree", "defense", 3, .09),
          ("cash", "economic", 5, .20), ("elite", "recruit_planter", 2, .04),
          ("elite", "tree_transfer", 3, .10), ("elite", "harvest_or_deliver", 3, .08),
          ("elite", "tactical", 4, .13), ("elite", "ordinary", 2, .06))


def elite_events(d):
    """Use consecutive same-seat rows. Worker slots can reorder each frame."""
    y = d["y"]
    rows = len(y)
    col = np.arange(rows)[:, None]
    nxt = np.minimum(np.arange(rows) + 1, rows - 1)[:, None]
    contiguous = (d["step"][nxt] == d["step"][:, None] + 1) & (d["gid"][nxt] == d["gid"][:, None])
    valid = free_slots(d)
    kind = d["tok_kind"][:, :y.shape[1]]
    pos = d["tok_pos"][:, :y.shape[1]].astype(np.intp)
    dest = DEST[pos, np.maximum(y, 0)]
    worker = valid & (kind < 2)
    here = d["maps"][col, 16, pos] > 0
    enemy_tree = d["maps"][col, 17, pos] > 0
    tree_after = d["maps"][nxt, 16, pos] > 0
    transfer = contiguous & worker & (kind == 1) & (y == 0) & ~here & tree_after

    # Match the next collector by both current and previous canonical cells.
    # A collector recruited this tick cannot satisfy the previous-cell test.
    post_pos = d["tok_pos"][nxt[:, 0], :10]
    post_prev = d["hist_cell"][nxt[:, 0], :10, -2]
    post_coll = (d["tok_kind"][nxt[:, 0], :10] == 0) & d["tok_mask"][nxt[:, 0], :10]
    matched = ((dest[:, :, None] == post_pos[:, None, :]) &
               (pos[:, :, None] == post_prev[:, None, :]) & post_coll[:, None, :])
    unique = matched.sum(-1) == 1
    post_cargo = (d["tok_cargo"][nxt[:, 0], :10][:, None, :] * matched).sum(-1)
    pre_cargo = d["tok_cargo"][:, :y.shape[1]]
    collector = contiguous & worker & (kind == 0) & unique
    on_base = d["maps"][col, 14, dest] > 0
    safe = d["maps"][col, 24, dest] == 0
    harvest = collector & (y == 0) & ~on_base & safe & (post_cargo > pre_cargo)
    delivery = collector & on_base & (pre_cargo > 0) & (post_cargo == 0)
    return dict(planter=valid & (kind == 1), recruit=valid & (kind == 2),
                recruit_planter=valid & (kind == 2) & (y == 2),
                new_tree=transfer & ~enemy_tree, seized_tree=transfer & enemy_tree,
                tree_transfer=transfer, harvest=harvest, delivery=delivery,
                harvest_or_deliver=harvest | delivery, ordinary=valid)


def load_elite(path, trainer):
    if sha(path) != ELITE_SHA:
        raise ValueError("unexpected top-five source identity")
    games = json.loads(path.read_bytes())["games"]
    rng = np.random.default_rng(SEED + 1)
    chunks = {"train": [], "val": []}
    for game in games:
        with checked_shard(path, game) as src:
            d = {k: src[k] for k in trainer.FIELDS}
        if not np.all(d["teacher"] == 2):
            raise ValueError("unexpected teacher style in elite data")
        events = elite_events(d)
        blob = (path.parent / game["mask_shard"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != game["mask_sha256"]:
            raise ValueError("top-five tactical mask identity mismatch")
        with np.load(io.BytesIO(blob), allow_pickle=False) as masks:
            valid = events["ordinary"]
            events["attack"] = masks["attack"] & valid
            events["defense"] = masks["defense"] & valid
        events["tactical"] = events["attack"] | events["defense"]
        # The validation sample is uniform by game and ignores labels. Training
        # includes up to eight extra rows per rare event to preserve real examples.
        take = 72 if game["split"] == "train" else 48
        chosen = rng.choice(len(d["y"]), min(take, len(d["y"])), replace=False)
        if game["split"] == "train":
            for key in ("new_tree", "seized_tree", "delivery", "recruit_planter", "attack", "defense"):
                eligible = np.flatnonzero(events[key].any(-1))
                if eligible.size:
                    extra = rng.choice(eligible, min(8, len(eligible)), replace=False)
                    chosen = np.union1d(chosen, extra)
        chosen = np.sort(chosen)
        chunks[game["split"]].append({**{k: d[k][chosen] for k in trainer.FIELDS},
                                      **{k: v[chosen] for k, v in events.items()}})
    return {s: {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}
            for s, ps in chunks.items()}, games


def load_tree(path, trainer):
    data, games, digest = load_games(path, trainer)
    expected = {"train": set(range(620124200, 620124218)), "val": set(range(620124218, 620124224))}
    if any({g["seed"] for g in games if g["split"] == s} != seeds for s, seeds in expected.items()):
        raise ValueError("the fixed 18/6 V3 game split is incomplete")
    for split, d in data.items():
        masks = {k: [] for k in TREE_MASKS}
        for g in games:
            if g["split"] != split:
                continue
            with checked_shard(path, g) as z:
                for key in TREE_MASKS:
                    m = z[key + "_mask"]
                    if m.shape != z["y"].shape or m.dtype != np.bool_ or np.any(m & (z["y"] < 0)):
                        raise ValueError(f"invalid {key} mask for {g['id']}")
                    masks[key].append(m)
        for key, parts in masks.items():
            d[key] = np.concatenate(parts) & free_slots(d)
        d["recruit_planter"] = d["recruit"] & (d["y"] == 2)
    return data, games, digest


@torch.inference_mode()
def evaluate(net, d, keys, trainer):
    net.eval()
    predicted, ce = [], []
    for j in range(0, len(d["y"]), 32):
        idx = np.arange(j, min(j + 32, len(d["y"])))
        b = trainer.to_dev(trainer.fetch(d, idx), aug=False)
        z = net(b)[0]
        if not torch.isfinite(z).all():
            raise ValueError("nonfinite validation logits")
        predicted.append(z.argmax(-1).cpu().numpy())
        ce.append(F.cross_entropy(z.transpose(1, 2), b["y"].clamp_min(0), reduction="none").cpu().numpy())
    pred, losses = np.concatenate(predicted), np.concatenate(ce)
    result = {}
    gids = np.broadcast_to(d["gid"][:, None], pred.shape)
    for k in keys:
        mask = d[k]
        result[k] = {**metrics(losses, pred == d["y"], mask), "games": int(len(np.unique(gids[mask]))),
                     "teacher_actions": np.bincount(d["y"][mask], minlength=5).tolist(),
                     "student_actions": np.bincount(pred[mask], minlength=5).tolist()}
    return result


def run(a):
    if a.out.exists() or a.receipt.exists() or a.out.resolve() == a.receipt.resolve():
        raise FileExistsError("fresh weight and receipt paths required")
    if any(k.startswith(("BC_", "CNET2_")) for k in os.environ):
        raise ValueError("unset BC_* and CNET2_* switches")
    os.environ["BC_DEVICE"] = a.device
    import cnet2_train as trainer

    started = time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    parent = a.init.read_bytes()
    if hashlib.sha256(parent).hexdigest() != PARENT_SHA or sha(a.cash) != CASH_SHA:
        raise ValueError("frozen V3 or rehearsal manifest identity mismatch")
    with np.load(io.BytesIO(parent), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        preset, cfg, count = resolve_preset_metadata(meta)
        if preset != "sapling-1.2m" or count != 1206476 or meta["training_status"] != "student_state_economic_correction":
            raise ValueError("wrong parent architecture or lineage")
        net = CNet2(**cfg)
        net.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k != "meta"})
    net.to(trainer.DEV)
    for name, p in net.named_parameters():
        if name.startswith(("value.", "utility.", "wdl.", "margin.", "aux.")):
            p.requires_grad_(False)

    elite, elite_games = load_elite(a.elite, trainer)
    cash, cash_games, _ = load_corrections(a.cash, trainer)
    for d in cash.values():
        d["economic"] = d["econ"]
    tree, tree_games, tree_sha = load_tree(a.tree, trainer)
    seeds = [g["seed"] for g in elite_games + cash_games + tree_games]
    if len(set(seeds)) != len(seeds):
        raise ValueError("source games share a map or training and validation overlap")
    data = {"cash": cash, "tree": tree, "elite": elite}
    val_keys = {"tree": (*TREE_MASKS, "recruit_planter"), "cash": ("economic",),
                "elite": ("ordinary", "new_tree", "seized_tree", "delivery", "harvest",
                         "recruit", "recruit_planter", "attack", "defense")}
    tasks = []
    for source, key, size, fraction in GROUPS:
        d = data[source]["train"]
        ids = np.flatnonzero(d[key].any(-1))
        used = source
        if not len(ids) and source == "tree" and key in ("planter", "recruit", "defense"):
            used, d = "elite", elite["train"]
            ids = np.flatnonzero(d[key].any(-1))
        if not len(ids):
            raise ValueError(f"no supervision for {source}/{key}")
        tasks.append((source, used, key, size, fraction, d, ids))
    def validation():
        return {src: evaluate(net, data[src]["val"], keys, trainer) for src, keys in val_keys.items()}

    before = validation()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    losses, slots = [], Counter()
    net.train()
    for step in range(STEPS):
        parts, per_slot = [], []
        for src, used, key, size, fraction, d, ids in tasks:
            idx = np.sort(rng.choice(ids, size, replace=len(ids) < size))
            parts.append(trainer.fetch(d, idx))
            selected = d[key][idx]
            slots[f"{src}/{key}"] += int(selected.sum())
            per_slot.append(selected.astype(np.float32) * (fraction / selected.sum()))
        b = trainer.to_dev({k: np.concatenate([p[k] for p in parts]) for k in trainer.LOAD}, aug=False)
        logits = net(b)[0]
        loss = (F.cross_entropy(logits.transpose(1, 2), b["y"].clamp_min(0), reduction="none") *
                torch.from_numpy(np.concatenate(per_slot)).to(trainer.DEV)).sum()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss.detach()))
        if (step+1) % 64 == 0:
            print(f"{step+1}/{STEPS} loss {losses[-1]:.4f}", flush=True)
    after = validation()
    trainer._check_finite_arrays({k: t.detach().cpu().numpy() for k, t in net.state_dict().items()}, "tree V4")
    recipe = dict(parent_sha256=PARENT_SHA, tree_manifest_sha256=tree_sha, cash_manifest_sha256=CASH_SHA,
                  elite_manifest_sha256=ELITE_SHA, trainer_sha256=sha(__file__), seed=SEED,
                  optimizer="AdamW", learning_rate=LR, weight_decay=.0001, grad_clip=5., updates=STEPS,
                  style=2, augmentation=False, non_policy_heads="no target, head parameters frozen; shared trunk trained",
                  candidate_selection="fixed final update", sources=[dict(source=src, used_source=used, mask=key,
                  roots_per_update=n, loss_weight=w, available_train_roots=len(ids),
                  available_train_slots=int(d[key].sum()), sampled_slots=slots[f"{src}/{key}"])
                  for src,used,key,n,w,d,ids in tasks])
    report = dict(schema="sapling-economic-tactical-bc/1", params=count, **recipe, before=before, after=after,
                  games={k:[dict(id=g["id"], seed=g["seed"], split=g["split"], sha256=g["sha256"])
                            for g in gs] for k, gs in (("cash",cash_games),("tree",tree_games),("elite",elite_games))},
                  loss=losses, runtime=dict(torch=str(torch.__version__), numpy=np.__version__, threads=a.threads,
                  device=str(trainer.DEV), seconds=time.monotonic()-started))
    for dest in (a.out, a.receipt):
        dest.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as td, tempfile.TemporaryDirectory(dir=a.receipt.parent) as rd:
        weight = Path(td) / "weights.npz"
        receipt = Path(rd) / "receipt.json"
        trainer.export(net, str(weight), dict(cfg=cfg, model_name="Sapling", preset=preset, params=count,
                       untrained=False, value_heads_untrained=True, training_status="tree_and_tactical_imitation",
                       tree_bc=recipe))
        report["trained_sha256"] = sha(weight)
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.link(receipt, a.receipt)
        try:
            os.link(weight, a.out)
        except BaseException:
            a.receipt.unlink()
            raise
    print(report["trained_sha256"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--cash", type=Path, required=True)
    p.add_argument("--elite", type=Path, required=True)
    p.add_argument("--init", type=Path, default=WEIGHT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--threads", default=2, type=int)
    args = p.parse_args()
    if args.threads < 1:
        p.error("positive --threads required")
    run(args)
