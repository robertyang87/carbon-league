#!/usr/bin/env python3
"""eval_shadow_ds -- fidelity + strength eval for shadow_deepseek.py.

Reuses eval_shadow_v2.play_game verbatim (same ledger instrumentation
and death attribution). Game list per the 2026-11-03 directives:
  * fidelity: shadow vs example, seed 2000, both sides (2 games)
  * strength: shadow vs base_v53, seeds 1000/1001, both sides (4 games,
    paired)
juzi is retired as an opponent (user directive 09:46) -- NOT played.

The caller is responsible for holding .eval_lock around this run.

Usage: python3 eval_shadow_ds.py SHADOW_FILE OUT_JSONL
"""

import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eval_shadow_v2 import play_game  # noqa: E402

GAMES = [
    (2000, 0, "example", "example.py"),
    (2000, 1, "example", "example.py"),
    (1000, 0, "base_v53", "base_v53.py"),
    (1000, 1, "base_v53", "base_v53.py"),
    (1001, 0, "base_v53", "base_v53.py"),
    (1001, 1, "base_v53", "base_v53.py"),
]


def main():
    shadow_file, out_path = sys.argv[1], sys.argv[2]
    out = open(out_path, "a")
    for seed, side, okind, ofile in GAMES:
        t0 = time.time()
        # both opponents (example, base_v53) draw from the global
        # random module; seed it exactly as eval_v53 does so games
        # are reproducible and trace replays match to 0.00
        random.seed(7919 * seed + 104729 * (1 - side) + 31)
        res = play_game(seed, side, okind, ofile, shadow_file)
        res["wall_s"] = round(time.time() - t0, 1)
        out.write(json.dumps(res) + "\n")
        out.flush()
        print(json.dumps(res), flush=True)
    out.close()


if __name__ == "__main__":
    main()
