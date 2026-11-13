#!/usr/bin/env python3
"""Case-C rollback verification: post-rollback fork_c_pricing.py
(price_soft=0, lethal+minreach on) re-run on the red-line breach
matchups + full mirror set. Reuses eval_c_pricing.play_game verbatim
(same harness/ledger/RNG protocol). Rows -> eval_c_verify_results.jsonl."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import eval_c_pricing  # noqa: E402

FORK = "fork_c_pricing.py"
JOBS = [("vshadow", FORK, "shadow_alpha_v2.py", 1000, 0),
        ("vshadow", FORK, "shadow_alpha_v2.py", 1003, 1)]
for _seed in (1000, 1001):
    for _side in (0, 1):
        JOBS.append(("vmirror", FORK, "base_v53.py", _seed, _side))

if __name__ == "__main__":
    out = open(os.path.join(HERE, "eval_c_verify_results.jsonl"), "a")
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(eval_c_pricing.play_game, t): t for t in JOBS}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["tag"], res["seed"], res["policy_side"],
                  res["my_cash"], res["opp_cash"], res["margin"],
                  "dc=", res["ledger_pol"]["death_cargo"], flush=True)
    out.close()
    print("VERIFY_DONE")
