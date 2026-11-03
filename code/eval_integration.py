"""eval_integration: paired-seed eval for the fork-v0.2 integration line.

Reuses eval_v53.play_game verbatim (same FastCarbon, same ledger, same
RNG protocol: random.seed(7919*seed + 104729*policy_side + 31), policy
= player A). Delta is read ONLY within this convention (GS-line lesson:
the seeding convention changes the game; cross-convention absolutes
are not comparable).

Usage:
  python3 eval_integration.py <batch> <a_file> <b_file> <seed_lo> <seed_hi>

Plays seeds seed_lo..seed_hi x sides {0,1} (A on both sides), at most
2 games concurrently, appends one JSON line per game (with "batch")
to eval_integration_results.jsonl.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_v53  # noqa: E402


def main():
    batch, a_file, b_file = sys.argv[1], sys.argv[2], sys.argv[3]
    lo, hi = int(sys.argv[4]), int(sys.argv[5])
    jobs = [(batch, a_file, b_file, seed, side)
            for seed in range(lo, hi + 1) for side in (0, 1)]
    out_path = os.path.join(HERE, "eval_integration_results.jsonl")
    with open(out_path, "a") as out:
        with ProcessPoolExecutor(max_workers=2) as ex:
            futs = {ex.submit(eval_v53.play_game, t): t for t in jobs}
            for fut in as_completed(futs):
                res = fut.result()
                res["batch"] = batch
                out.write(json.dumps(res) + "\n")
                out.flush()
                print(res["batch"], res["seed"], res["policy_side"],
                      res["my_cash"], res["opp_cash"], res["margin"],
                      f"a_ms={res['a_ms_per_step']}",
                      f"b_ms={res['b_ms_per_step']}", flush=True)


if __name__ == "__main__":
    main()
