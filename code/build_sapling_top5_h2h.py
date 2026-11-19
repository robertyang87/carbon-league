"""Extract both expert seats in frozen Top Five head-to-head ladder games for Sapling BC."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from build_sapling_elite_bc import ROOT, digest, write_json, STYLE_SLOT
import cnet2_build as builder
import cnet2_feat as F
from recover_public import canonical, validate

ARCHIVE = ROOT / "reports/master_family/tactical_rules"
RES = ROOT / "reports/sapling_1200k"
SELECTION = "sapling-top5-full-h2h-v2-20261119"
REACH = [set(int(i) for i in row) for row in F.DEST]
DESCRIPTIONS = {
    "attack": "Planter can reach an enemy tree (stay included), has seize cash before joint actions, and at least two turns of tree life and game remain.",
    "cargo_defense": "Positive-load collector staying is exposed to an equal/lighter enemy collector; a reachable alternative free of resident friendly workers and of this threat exists. Excludes enemy-base-adjacent roots.",
    "tree_defense": "A collector can reach a live own tree also reachable this step by an enemy planter with cash to seize; at least two turns of tree life and game remain.",
    "defense": "cargo_defense OR tree_defense.",
    "no_tactical_opportunity": "Valid expert slot with neither attack nor defense opportunity; normal economy and other decisions remain in BC."
}


def verified_game(k, row, record, replay_dir):
    path = replay_dir / (k + ".json.gz")
    blob = path.read_bytes()
    env = json.loads(gzip.decompress(blob))
    if digest(blob) != record["sha256"] and digest(canonical(env)) != record["canonical_sha256"]:
        raise ValueError(f"replay digest disagrees: {k}")
    check = validate(env, row)
    if check["environment_uuid"] != record["environment_uuid"] or check["step_count"] != record["steps"]:
        raise ValueError(f"replay metadata disagrees: {k}")
    return env, check


def slot_context(own, foe, cfg, turns_left, tok, seat, y, raw_actions):
    tree_life = cfg["treeLifespan"]
    can_raid = own[0] >= cfg["seizeCost"]
    can_be_raided = foe[0] >= cfg["seizeCost"]
    enemy_trees = {p for p, age in foe[3].values() if tree_life - age >= 2}
    own_trees = {p for p, age in own[3].values() if tree_life - age >= 2}
    foes_min = {}
    foe_planter_reach = set()
    for cell, cargo, role in foe[2].values():
        if role == "PLANTER":
            foe_planter_reach.update(REACH[cell])
        else:
            for q in REACH[cell]:
                foes_min[q] = min(foes_min.get(q, float("inf")), cargo)
    raids = (foe_planter_reach & own_trees) if can_be_raided else set()
    workers = {v[0]: (uid, v[1], v[2]) for uid, v in own[2].items()}
    bases = set(foe[1].values())
    # Every worker arrives on a unique occupied cell. Canonical tokens bind slot to UID.
    for s in range(10):
        if y[s] < 0:
            continue
        cell = int(F.CANON[tok[s]]) if seat else int(tok[s])
        uid, cargo, role = workers[cell]
        raw = raw_actions.get(uid, "NONE")
        actual = F.MOVES.index(raw)
        expected = F.CANON_MOVE[actual] if seat else actual
        if y[s] != expected:
            raise ValueError("expert command rewritten, refusing to treat repaired target as demonstration")
        # Eligibility uses only the pre-action player's public state. Use the logged
        # action solely to count the expert's response after assigning the mask.
        destination = int(F.DEST[cell, actual])
        reachable = REACH[cell]
        if role == "PLANTER":
            target = reachable & enemy_trees
            if turns_left >= 2 and can_raid and target:
                yield s, "attack", destination in target
            continue
        if turns_left >= 2 and raids & reachable:
            yield s, "tree_defense", destination in raids
        if cargo <= 0 or reachable & bases or foes_min.get(cell, float("inf")) > cargo:
            continue
        safe = {q for q in reachable if foes_min.get(q, float("inf")) > cargo}
        if any(q == cell or q not in workers for q in safe):
            yield s, "cargo_defense", destination in safe


def build_one(task, replay_dir, out):
    gid, g = task
    key = g["id"]
    path = replay_dir / (key + ".json.gz")
    env = json.loads(gzip.decompress(path.read_bytes()))
    game_margin = g["rewards"][0] - g["rewards"][1]
    shard = out / g["shard"]
    _, n, err = builder.build_game((str(path), [0, 1], gid, [STYLE_SLOT] * 2,
                                    [game_margin, -game_margin], str(shard)))
    T = len(env["steps"]) - 1
    if err or n != 2 * T:
        raise ValueError(f"incomplete BC shard {key}: rows={n}, {err}")
    masks = {name: np.zeros((n, F.OWN), dtype=bool) for name in ("attack", "cargo_defense", "tree_defense")}
    counts = Counter()
    with np.load(shard, allow_pickle=False) as ds:
        y, pos = ds["y"], ds["tok_pos"]
        if not np.array_equal(ds["step"], np.tile(np.arange(T), 2)):
            raise ValueError("slot mask order differs from builder timestep order")
        for seat in (0, 1):
            for t, frame in enumerate(env["steps"][:-1]):
                idx = seat * T + t
                own, foe = [frame[0]["observation"]["players"][p] for p in (seat, 1 - seat)]
                action = env["steps"][t + 1][seat]["action"] or {}
                remaining = env["configuration"]["episodeSteps"] - 1 - t
                for s, kind, responded in slot_context(own, foe, env["configuration"], remaining,
                                                        pos[idx], seat, y[idx], action):
                    masks[kind][idx, s] = True
                    counts[kind + "_expert_response"] += int(responded)
            for t in (0, T-1):
                if not (ds["teacher"][seat*T + t] == STYLE_SLOT):
                    raise ValueError("wrong inference style slot")
        masks["defense"] = masks["cargo_defense"] | masks["tree_defense"]
        if any(np.any(mask & (y < 0)) for mask in masks.values()):
            raise ValueError("opportunity mask addresses nonexistent worker")
        counts.update({name: int(np.count_nonzero(mask)) for name, mask in masks.items()})
        counts["labeled_slots"] = int((y >= 0).sum())
        counts["no_tactical_opportunity"] = int(((y >= 0) & ~(masks["attack"] | masks["defense"])).sum())
        counts["cargo_and_tree_defense"] = int((masks["cargo_defense"] & masks["tree_defense"]).sum())
    mask_path = out / g["mask_shard"]
    np.savez_compressed(mask_path, **masks)
    return dict(g, rows=n, sha256=digest(shard.read_bytes()), mask_sha256=digest(mask_path.read_bytes()),
                counts=dict(counts))


def prior_exclusions():
    elite_path = RES / "elite_bc_data.json"
    direct_path = RES / "elite_3s2_agreement.json"
    elite = json.loads(elite_path.read_bytes())["games"]
    direct = json.loads(direct_path.read_bytes())["games"]
    ids = {g["id"].split("-")[0] for g in elite + direct}
    seeds = {g["seed"] for g in elite} | {g["random_seed"] for g in direct}
    return ids, seeds, {p.name: digest(p.read_bytes()) for p in (elite_path, direct_path)}


def build(args):
    if any(k.startswith("CNET2_") for k in os.environ):
        raise ValueError("unset CNET2_* switches before the fixed build")
    raw = (ARCHIVE / "top5_inventory.json").read_bytes()
    inv = json.loads(raw)
    scores_raw = (ARCHIVE / inv["score_rows_file"]).read_bytes()
    if digest(scores_raw) != inv["score_rows_sha256"]:
        raise ValueError("frozen Top Five score archive changed")
    scores = json.loads(scores_raw)
    names = {r["name"]: r["id"] for r in inv["ranks"]}
    used_ids, used_seeds, excluded_docs = prior_exclusions()
    if len(names) != 5 or len(set(names.values())) != 5:
        raise ValueError("expected five distinct ladder registrations")
    used_seeds.update([620050000, 620050001, 620081731, *args.exclude_seed])
    inventory = []
    candidates = defaultdict(list)
    for k, s in sorted(scores.items()):
        if not (s["name_a"] in names and s["name_b"] in names and s["name_a"] != s["name_b"]):
            continue
        rec = inv["replays"][k]
        env, check = verified_game(k, s, rec, args.replays)
        seed = check["random_seed"]
        models = check["seat_model_names"]
        source = dict(id=k, models=models, strategy_ids=[names[m] for m in models], seed=seed,
                      canonical_sha256=rec["canonical_sha256"], git_blob=rec.get("git_blob"),
                      end_step=check["last_step"], rewards=check["rewards_by_seat"])
        reason = None
        if check["statuses"] != ["DONE", "DONE"]:
            reason = "execution_fault"
        elif s["id"] in used_ids or seed in used_seeds or 20291101 <= seed <= 20292100:
            reason = "excluded_match_or_map"
        if reason:
            inventory.append(dict(source, excluded=reason))
            continue
        if not isinstance(seed, int):
            raise ValueError(f"no engine random seed for {k}")
        used_ids.add(s["id"])
        used_seeds.add(seed)
        pair = tuple(sorted(models))
        candidates[pair].append(dict(source, shard=k+".npz", mask_shard=k+".masks.npz"))
        inventory.append(source)
    if len(inventory) != inv["shared_fixture_count"] or len(candidates) != 10:
        raise ValueError("incomplete head-to-head coverage")
    selected = []
    for pair, pool in sorted(candidates.items()):
        if len(pool) < 2:
            raise ValueError("cannot isolate validation for pair " + str(pair))
        pool.sort(key=lambda r: digest((SELECTION + "|" + r["id"]).encode()))
        n_val = max(1, round(0.2 * len(pool)))
        for i, g in enumerate(pool):
            selected.append(dict(g, split="val" if i < n_val else "train"))
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="sapling-top5-h2h/2", style_slot=STYLE_SLOT, selection_seed=SELECTION,
                    leaderboard_sha256=inv["leaderboard_raw_sha256"], inventory_sha256=digest(raw),
                    scores_sha256=digest(scores_raw), strategy_ids=names, excluded_docs=excluded_docs,
                    no_training_with_prior_games=True, extra_excluded_seeds=args.exclude_seed,
                    opportunity_definitions=DESCRIPTIONS, mutual_matches=inventory, games=selected)
    write_json(args.out / "selection.json", manifest)
    tasks = list(enumerate(selected))
    fn = partial(build_one, replay_dir=args.replays, out=args.out)
    if args.workers == 1:
        result = list(map(fn, tasks))
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            result = []
            for index, game in enumerate(pool.map(fn, tasks), 1):
                result.append(game)
                if index % 10 == 0:
                    print(f"verified {index}/{len(tasks)} games", flush=True)
    manifest["games"] = result
    for split in ("train", "val"):
        subset = [g for g in result if g["split"] == split]
        counts = Counter()
        for g in subset:
            counts.update(g["counts"])
        manifest[split] = dict(games=len(subset), rows=sum(g["rows"] for g in subset), **counts)
    write_json(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays", type=Path, required=True, help="restored top5_inventory.py compressed replays")
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    p.add_argument("--workers", type=int, default=2, choices=(1, 2, 3, 4))
    p.add_argument("--exclude-seed", type=int, action="append", default=[], help="extra held-out evaluation seeds")
    build(p.parse_args())
