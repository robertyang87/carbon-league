#!/usr/bin/env python3
"""distortion_check -- the checkpoint distortion checklist, computed
from recorded traces (never from agent self-reports).

Trace format (verified on disk): frames[i]["workers"] maps
"player-<p>-worker-<n>" -> [pos, cargo, "C"|"P"];
frames[i]["trees"] maps "player-<p>-tree-<n>" (creation id) ->
[pos, current_owner, age].  A steal keeps the tree id and flips the
owner field, so owner flips on one tid across consecutive frames are
steal events.

Metrics per game (distilled side = meta.policy_side):
  steals_by_policy / steals_from_policy, plants_by_policy,
  trees_owned_final, leftover_final / leftover_270 (cargo still on
  the policy's workers -- endgame cash-conversion failure shows here).

Usage: python3 distortion_check.py <trace.json.gz> [...]
"""

import gzip
import json
import sys


def owner_of_wid(wid):
    return int(wid.split("-")[1])


def carried(fr, side):
    return float(sum(w[1] for wid, w in fr["workers"].items()
                     if owner_of_wid(wid) == side))


def main():
    rows = []
    for path in sys.argv[1:]:
        with gzip.open(path, "rt") as f:
            tr = json.load(f)
        meta = tr["meta"]
        side = meta["policy_side"]
        frames = tr["frames"]
        steals_by = steals_from = plants = 0
        prev = {}
        for fr in frames:
            cur = {tid: t[1] for tid, t in fr["trees"].items()}
            for tid, owner in cur.items():
                if tid in prev:
                    if prev[tid] == 1 - side and owner == side:
                        steals_by += 1
                    elif prev[tid] == side and owner == 1 - side:
                        steals_from += 1
                elif owner == side:
                    plants += 1
            prev = cur
        last = frames[-1]
        rows.append({
            "file": path.split("/")[-1], "side": side,
            "final_cash": meta["final_cash"],
            "steals_by_policy": steals_by,
            "steals_from_policy": steals_from,
            "plants_by_policy": plants,
            "trees_owned_final": sum(
                1 for t in last["trees"].values() if t[1] == side),
            "leftover_final": carried(last, side),
            "leftover_270": carried(frames[270], side),
        })
        print(json.dumps(rows[-1], ensure_ascii=False))
    if rows:
        n = len(rows)
        print(json.dumps({
            "games": n,
            "steals_by_policy_avg": sum(r["steals_by_policy"]
                                        for r in rows) / n,
            "plants_by_policy_avg": sum(r["plants_by_policy"]
                                        for r in rows) / n,
            "leftover_final_avg": sum(r["leftover_final"]
                                      for r in rows) / n,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
