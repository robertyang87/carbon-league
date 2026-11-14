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


def _clustered_se(pairs):
    """SE of the mean with the SEED as the independent unit (section 20
    R2; same semantics as rl2_ppo_train._clustered_se): a seed's games
    (both seats) are averaged into one cluster value and the SE runs
    over clusters. Returns (se, n_clusters); se None with < 2 clusters."""
    by_seed = {}
    for seed, v in pairs:
        by_seed.setdefault(seed, []).append(float(v))
    if len(by_seed) < 2:
        return None, len(by_seed)
    cm = np.array([np.mean(v) for v in by_seed.values()])
    return float(cm.std(ddof=1) / math.sqrt(len(cm))), len(by_seed)


def summarize(by):
    """Per-opponent summary rows from {opp: [(seed, margin)]}. "se" is
    the seed-clustered SE; "se_iid" keeps the old per-game figure for
    continuity (it understates by up to sqrt(2) on paired seats).
    A single seed makes the clustered SE inestimable: the row emits
    se as null, never 0.0 (Codex re-review item 3)."""
    out = {}
    for o, pairs in by.items():
        m = np.array([v for _, v in pairs], dtype=np.float64)
        se, n_seeds = _clustered_se(pairs)
        out[o] = {"n": len(m), "n_seeds": n_seeds, "wins": int((m > 0).sum()),
                  "mean": round(float(m.mean()), 1),
                  "se": round(se, 1) if se is not None else None,
                  "se_iid": round(float(m.std(ddof=1) / math.sqrt(len(m))), 1) if len(m) > 1 else 0.0}
    return out


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
    if a.temp < 0:
        p.error("--temp must be >= 0 (0 = greedy)")
    if a.seed0 < 980000:
        print("warning: seeds < 980000 overlap other experiments' ranges", file=sys.stderr)
    # Greedy sends temperature 1.0, never 0.0: the adapters divide the
    # logits by the temperature BEFORE the argmax (rl2_ppo_cnet2 act:
    # z = lg / ts; the small adapter likewise), so a literal 0.0 poisons
    # every logit with inf/NaN and the "greedy" policy collapses to
    # argmax over NaNs (all-STAY). temp > 0 passes through verbatim.
    eff_temp = 1.0 if a.temp <= 0 else a.temp
    # V04: the evaluated net is not the trainer's learner. primary_role="eval"
    # keeps CNET2_TEMP_KIND (inherited from a trainer env) from silently
    # replacing the CLI temperature for sampling; the trainer's own tasks
    # omit the key and keep role="learner" (R04, by design).
    tasks = [{"adapter": a.adapter, "wkey": os.path.abspath(a.w), "games": [(o, a.seed0 + j, s)],
              "temp": eff_temp, "greedy": a.temp <= 0, "record": False, "rng_seed": j * 2 + s,
              "primary_role": "eval"}
             for o in a.opps.split(",") for j in range(a.pairs) for s in (0, 1)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(RO.play_batch, tasks, chunksize=1)
    by = {}
    for r in res:
        for g in r["games"]:
            by.setdefault(g["opp"], []).append((g.get("seed"), g["margin"]))
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    with open(a.log, "a") as f:
        for o, s in summarize(by).items():
            row = {"tag": a.tag, "w": a.w, "opp": o, **s,
                   "seed0": a.seed0, "wall_s": round(time.time() - t0), "t": time.time(),
                   # V04: the effective temperature is part of the reading's
                   # identity -- record what was asked, what was played, and
                   # whether a per-kind env was present (post-fix it no
                   # longer governs eval; the flag keeps the exposure
                   # visible in the row instead of silent).
                   "temp_cli": a.temp, "temp_eff": eff_temp, "greedy": a.temp <= 0,
                   "temp_kind_env": bool(os.environ.get("CNET2_TEMP_KIND"))}
            f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
