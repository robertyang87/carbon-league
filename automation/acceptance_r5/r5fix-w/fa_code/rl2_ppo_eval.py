"""rl2_ppo_eval -- standalone greedy paired evaluation of an rl2_ppo checkpoint.

One game per task (bit-faithful eval_v53 seeding), seeds SEED0..SEED0+N-1,
both sides. Appends summary rows to --log (default data/rl2/eval.jsonl).

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 \
  ../.venv/bin/python rl2_ppo_eval.py --adapter small --w ../data/rl2/x/ckpt_last.npz \
      --opps exp/opp_1cz_v5.py,exp/cand_c1z_v1.py --pairs 16
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BC_W", "64")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rl2_ppo_rollout as RO  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="small")
    p.add_argument("--w", required=True)
    p.add_argument("--opps", default="exp/opp_1cz_v5.py,exp/cand_c1z_v1.py")
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--seed0", type=int, default=980000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--temp", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--tag", default="")
    p.add_argument("--log", default=os.path.join(HERE, "..", "data", "rl2", "eval.jsonl"))
    a = p.parse_args()
    if a.seed0 < 980000:
        print("warning: seeds < 980000 overlap other experiments' ranges", file=sys.stderr)
    tasks = [{"adapter": a.adapter, "wkey": os.path.abspath(a.w), "games": [(o, a.seed0 + j, s)],
              "temp": max(a.temp, 1.0), "greedy": a.temp <= 0, "record": False, "rng_seed": j * 2 + s}
             for o in a.opps.split(",") for j in range(a.pairs) for s in (0, 1)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(RO.play_batch, tasks, chunksize=1)
    by = {}
    for r in res:
        for g in r["games"]:
            by.setdefault(g["opp"], []).append(g["margin"])
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    with open(a.log, "a") as f:
        for o, m in by.items():
            m = np.array(m)
            row = {"tag": a.tag, "w": a.w, "opp": o, "n": len(m), "wins": int((m > 0).sum()),
                   "mean": round(float(m.mean()), 1),
                   "se": round(float(m.std(ddof=1) / math.sqrt(len(m))), 1) if len(m) > 1 else 0.0,
                   "seed0": a.seed0, "wall_s": round(time.time() - t0), "t": time.time()}
            f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
