#!/usr/bin/env python3
"""extract_bc_ds -- DeepSeek-family (NaMnSi) BC extraction with the
corrected (obs_t -> act_{t+1}) pairing, over the UNION registry of
extract_bc_v2 (public snapshot + harvested accumulated, deduped by
match id, accumulated order derived from replay rewards).

Family = strategy name starts with "NaMnSi-DeepSeek" (single author).
The two ladder targets are NaMnSi-DeepSeek-RL-policy-1 (#2, 2578) and
NaMnSi-DeepSeek-RL-light-1 (#3, 2528) -- only 6 games each in the
corpus, so the corpus also includes the same author's NN-light sibling
series (capped at NN_CAP games per variant to bound featurization
cost) and the author's minor variants. Two npz files are written:

  bc_data/deepseek_family.npz   -- all family sides (capped)
  bc_data/deepseek_targets.npz  -- RL-policy-1 + RL-light-1 sides only

plus bc_data/deepseek_family_manifest.json. The misaligned
extract_bc.py pairing is NOT used; extraction reuses
extract_bc_fixed.extract_player_samples_fixed verbatim.

Run: python3 extract_bc_ds.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_bc as X  # noqa: E402
from extract_bc_fixed import extract_player_samples_fixed  # noqa: E402
from extract_bc_v2 import build_registry  # noqa: E402

REPLAYS = X.ROOT / "replays"
TARGETS = {"NaMnSi-DeepSeek-RL-policy-1", "NaMnSi-DeepSeek-RL-light-1"}
NN_CAP = 8  # max games per NN-light variant


def is_family(name):
    return name.startswith("NaMnSi-DeepSeek")


def main():
    reg = build_registry()

    fam_sides = []  # (meta, player)
    for meta in reg.values():
        for slot in ("a", "b"):
            if is_family(meta["strategies"][slot]["name"]):
                fam_sides.append((meta, X.player_of_slot(meta, slot)))
    per_strat = Counter(
        meta["strategies"][X.slot_of_player(meta, p)]["name"]
        for meta, p in fam_sides)
    print("family sides per strategy (games, pre-cap):")
    for name, n in per_strat.most_common():
        print(f"  {n:3d}  {name}")

    # cap NN-light variants
    seen = Counter()
    kept = []
    for meta, p in fam_sides:
        name = meta["strategies"][X.slot_of_player(meta, p)]["name"]
        if name.startswith("NaMnSi-DeepSeek-NN-light"):
            if seen[name] >= NN_CAP:
                continue
            seen[name] += 1
        kept.append((meta, p))
    print(f"kept {len(kept)} / {len(fam_sides)} sides after NN cap")

    counters = Counter()
    samples = []
    cache = {}
    for meta, player in kept:
        if meta["file"] not in cache:
            cache[meta["file"]] = X.load_game(REPLAYS / meta["file"])
        samples.extend(extract_player_samples_fixed(
            meta, cache[meta["file"]], player, counters))
    out = X.OUT_DIR / "deepseek_family.npz"
    np.savez_compressed(out, **X.pack(samples))
    print(f"deepseek_family: {len(samples)} samples; "
          f"counters={dict(counters)}")

    tgt = [s for s in samples if s["strategy"] in TARGETS]
    out2 = X.OUT_DIR / "deepseek_targets.npz"
    np.savez_compressed(out2, **X.pack(tgt))
    print(f"deepseek_targets: {len(tgt)} samples")

    pts = Counter(s["strategy"] for s in samples)
    games_seen = defaultdict(set)
    for s in samples:
        games_seen[s["strategy"]].add(s["file"])
    manifest = {"total_samples": len(samples), "strategies": {}}
    print("decision points per strategy:")
    for name, n in pts.most_common():
        print(f"  {n:6d} pts  {len(games_seen[name]):3d} games  {name}")
        manifest["strategies"][name] = {
            "decision_points": int(n),
            "games": len(games_seen[name]),
        }
    kept_ids = {meta["id"] for meta, _ in kept}
    manifest["games"] = [
        {"id": m["id"], "file": m["file"], "src": m["src"],
         "a": m["strategies"]["a"]["name"],
         "b": m["strategies"]["b"]["name"],
         "order": m["order"], "order_gap": m["order_gap"]}
        for m in reg.values() if m["id"] in kept_ids]
    with open(X.OUT_DIR / "deepseek_family_manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("wrote", out, out2, "+ manifest")


if __name__ == "__main__":
    main()
