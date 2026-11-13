"""eval_rl -- same-scale evaluation of rl_policy_v1 (rl-v0.1).

Reuses eval_v0_5.play_game verbatim (identical FastCarbon paired-seed
methodology and ledger instrumentation: deposits / tree_income /
recruit+plant+seize spend / deaths / death_cargo / stranded), writing
to eval_rl_results.jsonl instead.

Games:
  rl_policy_v1 vs reference_juzi, seeds 1000-1003, both sides (8)
  rl_policy_v1 vs policy_v0_5,    seed 1000, both sides (2)
  rl_policy_v1 vs example,        seed 2000, both sides (2)

Run: python3 eval_rl.py   (system python3, like the other evals)
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

from eval_v0_5 import play_game  # noqa: E402

TASKS = []
for seed in (1000, 1001, 1002, 1003):
    for side in (0, 1):
        TASKS.append((seed, side, "reference_juzi", "rl_policy_v1.py",
                      "rl_v01_juzi"))
for side in (0, 1):
    TASKS.append((1000, side, "policy_v0_5", "rl_policy_v1.py",
                  "rl_v01_v05"))
    TASKS.append((2000, side, "example", "rl_policy_v1.py",
                  "rl_v01_example"))


def main():
    out_path = os.path.join(HERE, "eval_rl_results.jsonl")
    out = open(out_path, "a")
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(play_game, t): t for t in TASKS}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["tag"], res["seed"], res["policy_side"],
                  res["my_cash"], res["opp_cash"], res["margin"],
                  flush=True)
    out.close()


if __name__ == "__main__":
    main()
