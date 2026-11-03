#!/usr/bin/env python3
"""extract_bc_fixed -- claude_family re-extraction with corrected
(obs, action) pairing.

WHY: the site's replay format stores in steps[t][player] the action
that PRODUCED steps[t]'s observation (it was decided from steps[t-1]'s
observation). Verified empirically on 4 Alpha games: worker movements
obs_t -> obs_{t+1} are predicted 100.0% by the action dict in
steps[t+1] and only ~40-45% by the one in steps[t]. extract_bc.py
paired (obs_t, act_t) -- every label shifted one step into the past.
This script pairs (obs_t, act_{t+1}) and writes
bc_data/claude_family_fixed.npz (original npz left untouched).

Everything else (canonical arrays, slot sorting, game selection)
reuses extract_bc.py's code verbatim.

Run: python3 extract_bc_fixed.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_bc as X  # noqa: E402


def extract_player_samples_fixed(game_meta, replay, player, counters):
    """Same as X.extract_player_samples but labels come from the NEXT
    entry's action dict (the decision actually made from this obs)."""
    slot = X.slot_of_player(game_meta, player)
    opp_slot = "b" if slot == "a" else "a"
    strategy = game_meta["strategies"][slot]["name"]
    opponent = game_meta["strategies"][opp_slot]["name"]
    score_self = float(game_meta["scores"][slot])
    score_opp = float(game_meta["scores"][opp_slot])
    won = game_meta.get("winner") == slot

    steps = replay["steps"]
    for t, entries in enumerate(steps):
        entry = entries[player]
        if entry.get("status") != "ACTIVE":
            counters["skipped_non_active"] += 1
            continue
        if t + 1 >= len(steps):
            counters["skipped_no_next"] += 1
            continue
        shared = entries[0]["observation"]
        own = entry["observation"]
        players = shared["players"]
        actions = steps[t + 1][player].get("action") or {}

        for uid in actions:
            if not uid.startswith(f"player-{player}-"):
                counters["action_key_mismatch"] += 1

        carbon = np.asarray(own["carbon"], dtype=np.float32)
        assert carbon.shape == (225,), carbon.shape

        cash = np.zeros(2, dtype=np.float32)
        bases = np.full((2, X.MAX_BASES), -1, dtype=np.int16)
        workers = np.zeros((2, X.MAX_WORKERS, 3), dtype=np.float32)
        workers_mask = np.zeros((2, X.MAX_WORKERS), dtype=bool)

        sorted_worker_ids = {}
        for i in range(2):
            p_cash, p_bases, p_workers, _p_trees = players[i]
            cash[i] = float(p_cash)
            for j, cid in enumerate(
                    sorted(p_bases, key=X.num_suffix)[:X.MAX_BASES]):
                bases[i, j] = int(p_bases[cid])
            wids = sorted(p_workers, key=X.num_suffix)
            sorted_worker_ids[i] = wids
            for j, wid in enumerate(wids[:X.MAX_WORKERS]):
                pos, cargo, wtype = p_workers[wid]
                workers[i, j, 0] = float(pos)
                workers[i, j, 1] = float(cargo)
                workers[i, j, 2] = float(X.WORKER_TYPE[wtype])
                workers_mask[i, j] = True
            if len(wids) > X.MAX_WORKERS:
                counters["workers_overflow"] += 1

        tree_rows = []
        for i in range(2):
            for tid, (pos, age) in players[i][3].items():
                tree_rows.append(
                    (i, X.num_suffix(tid), float(pos), float(age)))
        tree_rows.sort()
        if len(tree_rows) > X.MAX_TREES:
            counters["trees_overflow"] += 1
        trees = np.zeros((X.MAX_TREES, 3), dtype=np.float32)
        trees_mask = np.zeros(X.MAX_TREES, dtype=bool)
        for j, (owner, _n, pos, age) in enumerate(tree_rows[:X.MAX_TREES]):
            trees[j] = (pos, age, float(owner))
            trees_mask[j] = True

        worker_actions = np.zeros(X.MAX_WORKERS, dtype=np.int8)
        for j, wid in enumerate(sorted_worker_ids[player][:X.MAX_WORKERS]):
            act = actions.get(wid)
            if act is None:
                continue
            if act in X.WORKER_ACT:
                worker_actions[j] = X.WORKER_ACT[act]
            else:
                counters["unknown_worker_action"] += 1
        base_actions = np.zeros(X.MAX_BASES, dtype=np.int8)
        my_bases = players[player][1]
        for j, cid in enumerate(
                sorted(my_bases, key=X.num_suffix)[:X.MAX_BASES]):
            act = actions.get(cid)
            if act is None:
                continue
            if act in X.BASE_ACT:
                base_actions[j] = X.BASE_ACT[act]
            else:
                counters["unknown_base_action"] += 1

        yield {
            "step": int(shared.get("step", t)),
            "player": player,
            "carbon": carbon,
            "cash": cash,
            "bases": bases,
            "workers": workers,
            "workers_mask": workers_mask,
            "trees": trees,
            "trees_mask": trees_mask,
            "worker_actions": worker_actions,
            "base_actions": base_actions,
            "file": game_meta["file"],
            "strategy": strategy,
            "opponent": opponent,
            "score_self": score_self,
            "score_opp": score_opp,
            "won": won,
        }


def main():
    import json
    index = json.loads(X.INDEX_PATH.read_text(encoding="utf-8"))
    claude_games = []
    for g in index:
        for slot in ("a", "b"):
            if X.CLAUDE_RE.search(g["strategies"][slot]["name"]):
                claude_games.append((g, X.player_of_slot(g, slot)))
    counters = Counter()
    samples = []
    cache = {}
    for meta, player in claude_games:
        if meta["file"] not in cache:
            cache[meta["file"]] = X.load_game(
                X.ROOT / "replays" / meta["file"])
        samples.extend(extract_player_samples_fixed(
            meta, cache[meta["file"]], player, counters))
    out = X.OUT_DIR / "claude_family_fixed.npz"
    np.savez_compressed(out, **X.pack(samples))
    print(f"claude_family_fixed: {len(claude_games)} games -> "
          f"{len(samples)} samples; counters={dict(counters)}")
    print("wrote", out)


if __name__ == "__main__":
    main()
