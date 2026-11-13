#!/usr/bin/env python3
"""extract_bc_v53 -- build bc_data/bc_v53_corpus.npz from corpus-batch-1
traces (base_v53 self-play / sparring games recorded by
corpus_batch1_run.py).

Pairing (the whole point): trace frames follow the site-replay
convention -- frames[t+1]['acts'] holds the actions that PRODUCED
frames[t+1]'s state, i.e. the decisions made FROM frames[t]'s state.
Samples therefore pair frames[t] (observation) with
frames[t+1]['acts'] (labels): the corrected (obs_t -> act_{t+1})
pairing of extract_bc_fixed.py, NOT the same-frame pairing.
Validated on this corpus itself (seed-2000 mirror game): corrected
pairing predicts worker displacements 2703/2703 = 100.0%, uncorrected
control 37.0%.

Visible carbon: trace frames store the FULL carbon map, while the BC
convention (claude_family_fixed.npz) stores the acting player's
visible carbon.  It is rebuilt from the frame state with the env's own
rule (fastsim.obs): a collector sees its own cell, a planter sees its
4 toroidal neighbour cells (not its own); everything else 0.
Validated byte-exact against live sim.obs() carbon captured during
the smoke game at states 0/50/150/299 for both players.

Fields are packed with extract_bc.pack, so the npz is key-for-key and
dtype-for-dtype identical to claude_family_fixed.npz and the two can
be concatenated directly.  Only v53-family sides are collected as
teachers (manifest 'teacher_players'): base_v53 on either side,
fork_v0_1 on its side; GloriousEvolution / shadow / v0.5 sides are
opponents, never labels.

Known inherited precision: worker cargo in traces is rounded to 2 dp
(record_trace.snapshot), cell carbon to 3 dp (env native).

Run: python3 extract_bc_v53.py
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_bc as X  # noqa: E402

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "corpus_batch1_manifest.json"
TRACES = HERE / "traces"
OUT = X.OUT_DIR / "bc_v53_corpus.npz"
STATS_OUT = HERE / "corpus_batch1_extract_stats.json"
SIZE = 15


def visible_carbon(frame, player):
    vis = np.zeros(SIZE * SIZE, dtype=np.float64)
    full = frame["carbon"]
    for wid, (pos, _cargo, wtype) in frame["workers"].items():
        if not wid.startswith(f"player-{player}-"):
            continue
        if wtype == "C":
            vis[pos] = full[pos]
        else:
            x, y = divmod(int(pos), SIZE)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = ((x + dx) % SIZE) * SIZE + (y + dy) % SIZE
                vis[nb] = full[nb]
    return vis.astype(np.float32)


def samples_from_game(meta_game, trace, counters):
    frames = trace["frames"]
    meta = trace["meta"]
    final_cash = meta["final_cash"]
    bases_meta = meta["bases"]
    out = []
    for p in meta_game["teacher_players"]:
        opp = 1 - p
        strategy = meta_game["names"][str(p)]
        opponent = meta_game["names"][str(opp)]
        score_self = float(final_cash[p])
        score_opp = float(final_cash[opp])
        won = score_self > score_opp

        # per-player sorted base ids (stable across the whole game)
        sorted_bases = {}
        for i in range(2):
            rows = sorted(bases_meta[str(i)],
                          key=lambda r: X.num_suffix(r[0]))
            sorted_bases[i] = rows

        for t in range(300):
            frame = frames[t]
            labels = frames[t + 1]["acts"][p]

            cash = np.asarray(frame["cash"], dtype=np.float32)
            bases = np.full((2, X.MAX_BASES), -1, dtype=np.int16)
            for i in range(2):
                for j, (_bid, pos) in enumerate(
                        sorted_bases[i][:X.MAX_BASES]):
                    bases[i, j] = int(pos)

            workers = np.zeros((2, X.MAX_WORKERS, 3), dtype=np.float32)
            workers_mask = np.zeros((2, X.MAX_WORKERS), dtype=bool)
            sorted_wids = {}
            for i in range(2):
                wids = sorted(
                    (w for w in frame["workers"]
                     if w.startswith(f"player-{i}-")),
                    key=X.num_suffix)
                sorted_wids[i] = wids
                if len(wids) > X.MAX_WORKERS:
                    counters["workers_overflow"] += 1
                for j, wid in enumerate(wids[:X.MAX_WORKERS]):
                    pos, cargo, wtype = frame["workers"][wid]
                    workers[i, j, 0] = float(pos)
                    workers[i, j, 1] = float(cargo)
                    workers[i, j, 2] = float(
                        X.WORKER_TYPE["COLLECTOR" if wtype == "C"
                                      else "PLANTER"])
                    workers_mask[i, j] = True

            tree_rows = []
            for tid, (pos, owner, age) in frame["trees"].items():
                tree_rows.append(
                    (int(owner), X.num_suffix(tid), float(pos),
                     float(age)))
            tree_rows.sort()
            if len(tree_rows) > X.MAX_TREES:
                counters["trees_overflow"] += 1
            trees = np.zeros((X.MAX_TREES, 3), dtype=np.float32)
            trees_mask = np.zeros(X.MAX_TREES, dtype=bool)
            for j, (owner, _n, pos, age) in enumerate(
                    tree_rows[:X.MAX_TREES]):
                trees[j] = (pos, age, float(owner))
                trees_mask[j] = True

            worker_actions = np.zeros(X.MAX_WORKERS, dtype=np.int8)
            for j, wid in enumerate(sorted_wids[p][:X.MAX_WORKERS]):
                act = labels.get(wid)
                if act is None:
                    continue
                if act in X.WORKER_ACT:
                    worker_actions[j] = X.WORKER_ACT[act]
                else:
                    counters["unknown_worker_action"] += 1
            base_actions = np.zeros(X.MAX_BASES, dtype=np.int8)
            for j, (bid, _pos) in enumerate(
                    sorted_bases[p][:X.MAX_BASES]):
                act = labels.get(bid)
                if act is None:
                    continue
                if act in X.BASE_ACT:
                    base_actions[j] = X.BASE_ACT[act]
                else:
                    counters["unknown_base_action"] += 1

            out.append({
                "step": t,
                "player": p,
                "carbon": visible_carbon(frame, p),
                "cash": cash,
                "bases": bases,
                "workers": workers,
                "workers_mask": workers_mask,
                "trees": trees,
                "trees_mask": trees_mask,
                "worker_actions": worker_actions,
                "base_actions": base_actions,
                "file": meta_game["file"],
                "strategy": strategy,
                "opponent": opponent,
                "score_self": score_self,
                "score_opp": score_opp,
                "won": won,
            })
    return out


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counters = Counter()
    samples = []
    per_matchup = Counter()
    per_game_points = {}
    for g in manifest:
        with gzip.open(TRACES / g["file"], "rt", encoding="utf-8") as fh:
            trace = json.load(fh)
        got = samples_from_game(g, trace, counters)
        samples.extend(got)
        per_matchup[g["tag"]] += len(got)
        per_game_points[g["file"]] = len(got)
    packed = X.pack(samples)
    np.savez_compressed(OUT, **packed)
    print(f"games={len(manifest)} samples={len(samples)} "
          f"counters={dict(counters)}")
    print("per matchup:", dict(per_matchup))
    print("wrote", OUT)

    # stats for the data card
    arr = packed
    stats = {
        "games": len(manifest),
        "samples": len(samples),
        "per_matchup_samples": dict(per_matchup),
        "per_strategy": dict(Counter(arr["strategy"].tolist())),
        "per_opponent": dict(Counter(arr["opponent"].tolist())),
        "counters": dict(counters),
        "step_bands": {
            "opening_0_49": int(((arr["step"] >= 0)
                                 & (arr["step"] < 50)).sum()),
            "treewar_50_199": int(((arr["step"] >= 50)
                                   & (arr["step"] < 200)).sum()),
            "endgame_200_299": int((arr["step"] >= 200).sum()),
        },
        "teacher_won_share": float(arr["won"].mean()),
    }
    # action distributions over valid slots only (extract_bc convention)
    wdist, bdist = Counter(), Counter()
    for i in range(len(samples)):
        p = int(arr["player"][i])
        valid = arr["workers_mask"][i][p]
        for a, v in zip(arr["worker_actions"][i], valid):
            if v:
                wdist[int(a)] += 1
        nb = int((arr["bases"][i][p] >= 0).sum())
        for a in arr["base_actions"][i][:nb]:
            bdist[int(a)] += 1
    stats["worker_action_dist"] = {str(k): v for k, v in
                                   sorted(wdist.items())}
    stats["base_action_dist"] = {str(k): v for k, v in
                                 sorted(bdist.items())}
    STATS_OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
