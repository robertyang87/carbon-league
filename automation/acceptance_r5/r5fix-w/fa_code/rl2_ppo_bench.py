"""rl2_ppo_bench -- rollout throughput of rl2_ppo workers (recording on, sampling).

  ../.venv/bin/python rl2_ppo_bench.py --adapter cnet2 --w W.npz --workers 8 --games 32 --lockstep 2 \
      --opps exp/opp_1cz_v5.py,exp/cand_c1z_v1.py
Prints games/min, decisions/s, per-game policy/opponent seconds and worker peak RSS.
"""

import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BC_W", "64")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rl2_ppo_rollout as RO  # noqa: E402


def _task(t):
    r = RO.play_batch(t)
    r["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    r["rec_mb"] = sum(sum(sum(v.nbytes for v in rec.values() if hasattr(v, "nbytes")) for rec in g["recs"])
                      for g in r["games"]) / 1e6
    for g in r["games"]:
        g.pop("recs", None)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="cnet2")
    p.add_argument("--w", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--lockstep", type=int, default=2)
    p.add_argument("--opps", default="exp/opp_1cz_v5.py,exp/cand_c1z_v1.py")
    a = p.parse_args()
    opps = a.opps.split(",")
    specs = [(opps[i % len(opps)], 990500 + i // 2, i % 2) for i in range(a.games)]
    tasks = [{"adapter": a.adapter, "wkey": os.path.abspath(a.w), "games": specs[i:i + a.lockstep], "temp": 1.0,
              "greedy": False, "record": True, "rng_seed": i} for i in range(0, len(specs), a.lockstep)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        t0 = time.time()
        res = pool.map(_task, tasks, chunksize=1)
        wall = time.time() - t0
    ng = sum(len(r["games"]) for r in res)
    nd = sum(r["ndec"] for r in res)
    out = {"adapter": a.adapter, "workers": a.workers, "lockstep": a.lockstep, "games": ng, "wall_s": round(wall, 1),
           "games_per_min": round(60 * ng / wall, 1), "dec_per_s": round(nd / wall, 1),
           "pol_s_per_game": round(sum(r["t_pol"] for r in res) / ng, 2),
           "opp_s_per_game": round(sum(r["t_opp"] for r in res) / ng, 2),
           "sim_s_per_game": round(sum(r["t_sim"] for r in res) / ng, 2),
           "worker_maxrss_mb": round(max(r["maxrss_mb"] for r in res)),
           "rec_mb_per_game": round(sum(r["rec_mb"] for r in res) / ng, 2),
           "load": os.getloadavg()}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
