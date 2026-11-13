#!/usr/bin/env python3
"""profile_ds_replays -- ground-truth behaviour profile of the two
ladder targets (NaMnSi-DeepSeek-RL-policy-1 / -light-1) computed
directly from their replay files, for the shadow fidelity comparison.

Per target game and target player, from the replay steps' shared
players blocks:
  * final score (index scores, cross-checked vs replay rewards)
  * step-200 composition (collectors / planters / own trees)
  * plants (new tree ids first owned by the player), seizes (tree ids
    whose owner flips to the player), final tree count
  * deaths (own worker ids vanishing between consecutive steps) and
    the cargo they carried (median reported) -- replay-side death
    cargo is an upper-bound proxy: a vanished worker's cargo may in
    part be consolidated by a surviving own collector on the same
    cell; the sim ledger used in eval games is the exact figure.
  * stranded cargo at the final step (sum of own workers' cargo)

Run: python3 profile_ds_replays.py
Writes profile_ds_replays.json and prints a summary.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_bc as X  # noqa: E402
from extract_bc_v2 import build_registry  # noqa: E402

TARGETS = ("NaMnSi-DeepSeek-RL-policy-1", "NaMnSi-DeepSeek-RL-light-1")
REPLAYS = X.ROOT / "replays"


def profile_game(meta, player):
    replay = X.load_game(REPLAYS / meta["file"])
    steps = replay["steps"]
    slot = X.slot_of_player(meta, player)
    score = float(meta["scores"][slot])

    def blocks(t):
        return steps[t][0]["observation"]["players"]

    comp200 = None
    plants = seizes = 0
    prev_trees = {}  # tid -> owner
    deaths = 0
    death_cargo = []
    prev_workers = {}  # wid -> cargo
    final_stranded = None
    for t in range(len(steps)):
        pl = blocks(t)
        trees = {}
        for q in range(2):
            for tid in pl[q][3]:
                trees[tid] = q
        if t > 0:
            for tid, q in trees.items():
                if q != player:
                    continue
                if tid not in prev_trees:
                    plants += 1
                elif prev_trees[tid] != player:
                    seizes += 1
        prev_trees = trees
        workers = {wid: float(w[1]) for wid, w in pl[player][2].items()}
        if t > 0:
            for wid, cargo in prev_workers.items():
                if wid not in workers:
                    deaths += 1
                    death_cargo.append(cargo)
        prev_workers = workers
        if steps[t][0]["observation"].get("step", t) == 200 or t == 200:
            ws = pl[player][2]
            comp200 = {
                "collectors": sum(1 for w in ws.values()
                                  if w[2] == "COLLECTOR"),
                "planters": sum(1 for w in ws.values()
                                if w[2] == "PLANTER"),
                "trees": sum(1 for q in trees.values() if q == player),
            }
        if t == len(steps) - 1:
            final_stranded = round(sum(workers.values()), 1)
    return {
        "file": meta["file"], "slot": slot,
        "strategy": meta["strategies"][slot]["name"],
        "opponent": meta["strategies"]["b" if slot == "a" else "a"]["name"],
        "score": score, "comp200": comp200, "plants": plants,
        "seizes": seizes, "deaths": deaths,
        "death_cargo_median": (round(statistics.median(death_cargo), 1)
                               if death_cargo else None),
        "death_cargo_total_proxy": round(sum(death_cargo), 1),
        "final_trees": sum(1 for q in prev_trees.values() if q == player),
        "stranded": final_stranded,
    }


def main():
    reg = build_registry()
    rows = []
    for meta in reg.values():
        for slot in ("a", "b"):
            if meta["strategies"][slot]["name"] in TARGETS:
                rows.append(profile_game(
                    meta, X.player_of_slot(meta, slot)))
    by = defaultdict(list)
    for r in rows:
        by[r["strategy"]].append(r)
    out = {"games": rows, "summary": {}}
    for name, rs in by.items():
        s = {
            "games": len(rs),
            "score_mean": round(statistics.mean(r["score"] for r in rs), 1),
            "score_median": round(
                statistics.median(r["score"] for r in rs), 1),
            "plants_mean": round(statistics.mean(r["plants"] for r in rs), 1),
            "seizes_mean": round(statistics.mean(r["seizes"] for r in rs), 1),
            "deaths_mean": round(statistics.mean(r["deaths"] for r in rs), 1),
            "final_trees_mean": round(
                statistics.mean(r["final_trees"] for r in rs), 1),
            "stranded_mean": round(
                statistics.mean(r["stranded"] for r in rs), 1),
        }
        c200 = [r["comp200"] for r in rs if r["comp200"]]
        if c200:
            for k in ("collectors", "planters", "trees"):
                s[f"comp200_{k}_mean"] = round(
                    statistics.mean(c[k] for c in c200), 1)
        out["summary"][name] = s
        print(name, json.dumps(s, ensure_ascii=False))
    with open("profile_ds_replays.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote profile_ds_replays.json with", len(rows), "game rows")


if __name__ == "__main__":
    main()
