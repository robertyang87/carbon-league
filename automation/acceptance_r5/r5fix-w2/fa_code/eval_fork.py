"""eval_fork: paired-seed eval for the v53-fork tuning line.

Reuses eval_v53.play_game verbatim (same FastCarbon, same RNG protocol
random.seed(7919*seed + 104729*side + 31), same ledger reconstruction),
so every number is directly comparable with eval_v53_results.jsonl and
with BASELINE.md. Results append to eval_fork_results.jsonl.

Phases (argv[1]):
  scan     rho tiers {0.04,0.06,0.08,0.10} vs base_v53,
           seeds 1000-1001 x2 sides (4 games per tier, 16 total)
  shadow   base_v53 vs shadow_alpha_v2, seed 1000 x2 (structure check)
  confirm  two tiers given as argv[2], argv[3] (file stems, e.g.
           fork_rho006), seeds 1002-1003 x2 sides vs base (extends
           those tiers to 8 games)
  forkshadow fork_v0_1 vs shadow_alpha_v2, seeds 1000-1001 x2
           (4 games, NN-type reference; juzi retired as an opponent
           by user directive 2026-11-03 09:46)
  forkbase fork_v0_1 vs base_v53, seeds 1000-1003 x2 (8 games; only
           needed if the fork is not identical to a scanned tier)
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_v53  # noqa: E402

BASE = "base_v53.py"
JUZI = "reference_juzi.py"

TIERS = {0.04: "fork_rho004.py", 0.06: "fork_rho006.py",
         0.08: "fork_rho008.py", 0.10: "fork_rho010.py"}
TAGS = {"fork_rho004.py": "rho004", "fork_rho006.py": "rho006",
        "fork_rho008.py": "rho008", "fork_rho010.py": "rho010"}


def jobs_for(phase, extra):
    if phase == "scan":
        return [(TAGS[f], f, BASE, s, side)
                for f in TIERS.values()
                for s in (1000, 1001) for side in (0, 1)]
    if phase == "shadow":
        return [("v53shadow", BASE, "shadow_alpha_v2.py", 1000, side)
                for side in (0, 1)]
    if phase == "confirm":
        return [(TAGS[f"{stem}.py"], f"{stem}.py", BASE, s, side)
                for stem in extra
                for s in (1002, 1003) for side in (0, 1)]
    if phase == "forkshadow":
        return [("forkshadow", "fork_v0_1.py", "shadow_alpha_v2.py",
                 s, side)
                for s in (1000, 1001) for side in (0, 1)]
    if phase == "forkbase":
        return [("forkbase", "fork_v0_1.py", BASE, s, side)
                for s in (1000, 1001, 1002, 1003) for side in (0, 1)]
    raise SystemExit(f"unknown phase {phase}")


def main():
    phase = sys.argv[1]
    jobs = jobs_for(phase, sys.argv[2:])
    out = open(os.path.join(HERE, "eval_fork_results.jsonl"), "a")
    with ProcessPoolExecutor(max_workers=int(os.environ.get("EVAL_WORKERS", "2"))) as ex:
        futs = {ex.submit(eval_v53.play_game, t): t for t in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["tag"], res["seed"], res["policy_side"],
                  res["my_cash"], res["opp_cash"], res["margin"],
                  f"a_ms={res['a_ms_per_step']} b_ms={res['b_ms_per_step']}",
                  flush=True)
    out.close()


if __name__ == "__main__":
    main()
