#!/usr/bin/env python3
"""Split roots_calib.json into per-game shard manifests.

Usage: build_duel_shards.py TAG
Writes /home/arm/junyang/k0_calib/shards/TAG/game-<game>.json (198 roots each)
and /home/arm/junyang/k0_calib/shards/TAG/index.json (game order + counts).
Idempotent: existing shard files with identical content are left untouched.
"""
import json, os, sys

BASE = "/home/arm/junyang/k0_calib"
MANIFEST = os.path.join(BASE, "roots_calib.json")

def main():
    tag = sys.argv[1]
    outdir = os.path.join(BASE, "shards", tag)
    os.makedirs(outdir, exist_ok=True)
    entries = json.load(open(MANIFEST))
    by = {}
    order = []
    for e in entries:
        g = e["game"]
        if g not in by:
            by[g] = []
            order.append(g)
        by[g].append(e)
    assert len(order) == 50 and all(len(by[g]) == 198 for g in order), \
        "expected 50 games x 198 roots, got %d games" % len(order)
    index = {"tag": tag, "games": order, "roots_per_game": 198,
             "total_roots": len(entries), "total_games": len(order)}
    for g in order:
        p = os.path.join(outdir, "game-" + g + ".json")
        blob = json.dumps(by[g], sort_keys=True)
        if os.path.exists(p) and open(p).read() == blob:
            continue
        with open(p, "w") as fh:
            fh.write(blob)
    with open(os.path.join(outdir, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    print("SHARDS %s: %d games x 198 roots = %d roots" % (tag, len(order), len(entries)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
