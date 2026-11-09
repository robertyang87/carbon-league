"""awr_weights DS CORPUS_DIR [CORPUS_DIR ...] -- AWR weights on a build_awr dataset.

Reward (recomputed from each replay's final state at step 100, our side minus theirs):
  cash + cargo + sum over standing trees of  4-neighbour carbon * (1 - 0.9625^min(50 - age, 30)) * p_survive,
  p_survive = 0.3 if an enemy planter stands within 1 step of the tree (it is about to be seized), else 0.8
  (2026-11-09 review: a reward taken at the step-100 instant overvalues trees planted for the opponent to take).
Advantage: leave-one-out baseline over all games of the same seed (the greedy game and the K sampled ones):
  A_i = R_i - mean(R_j, j != i)   (lower variance than "minus the greedy game" alone).
Weights: w = exp(A / beta); beta starts at std(A) and grows until the effective sample size
  (sum w)^2 / sum w^2 is at least ESS_MIN (default 20%) of the games -- otherwise a few lucky games dominate.
uw (per unit) = 1 for planters and base slots, 0 for our collectors (cars stay the net's own)."""
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np


def side_value(obs, carbon, p):
    P = obs["players"]
    cash = float(P[p][0])
    cargo = sum(float(w[1]) for w in P[p][2].values())
    epl = [int(w[0]) for w in P[1 - p][2].values() if w[2] == "PLANTER"]
    v = 0.0
    for pos, age in P[p][3].values():
        pos = int(pos); r, c = divmod(pos, 15)
        nb = sum(carbon[((r + dr) % 15) * 15 + (c + dc) % 15] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        near = any(min(abs(e // 15 - r), 15 - abs(e // 15 - r)) + min(abs(e % 15 - c), 15 - abs(e % 15 - c)) <= 1 for e in epl)
        v += nb * (1.0 - 0.9625 ** max(0, min(50 - int(age), 30))) * (0.3 if near else 0.8)
    return cash + cargo + v


def rescore(corpus):
    games = [json.loads(l) for l in open(os.path.join(corpus, "awr.jsonl"))]
    for g in games:
        st = json.load(gzip.open(os.path.join(corpus, g["file"])))["steps"]
        o = st[-1][0]["observation"]; carbon = o.get("full_carbon") or o["carbon"]
        g["reward2"] = side_value(o, carbon, g["side"]) - side_value(o, carbon, 1 - g["side"])
        g["corpus"] = os.path.basename(corpus.rstrip("/"))
    by = defaultdict(list)
    for g in games:
        by[g["seed"]].append(g)
    for gs in by.values():
        tot = sum(x["reward2"] for x in gs)
        for x in gs:
            x["adv2"] = x["reward2"] - (tot - x["reward2"]) / max(1, len(gs) - 1)
    return games


def main():
    ds, corpora = sys.argv[1], sys.argv[2:]
    games = [g for c in corpora for g in rescore(c)]
    A = np.array([g["adv2"] for g in games])
    ess_min = float(os.environ.get("AWR_ESS_MIN", "0.2")); wmax = float(os.environ.get("AWR_WMAX", "20"))
    beta = float(A.std()) or 1.0
    for _ in range(40):
        w_ = np.minimum(wmax, np.exp(A / beta))
        ess = w_.sum() ** 2 / (w_ ** 2).sum() / len(w_)
        if ess >= ess_min:
            break
        beta *= 1.25
    wkey = {g["key"]: float(min(wmax, np.exp(g["adv2"] / beta))) for g in games}
    meta = json.load(open(os.path.join(ds, "meta.json"))); gid = np.load(os.path.join(ds, "gid.npy"))
    gw = np.array([wkey[g["key"].replace(".json.gz", "")] for g in meta["games"]], np.float32)
    w = gw[gid].astype(np.float32)
    kind = np.load(os.path.join(ds, "tok_kind.npy"), mmap_mode="r")
    S = np.load(os.path.join(ds, "y.npy"), mmap_mode="r").shape[1]
    uw = np.ones((len(gid), S), np.float32)
    uw[:, :10] = (np.asarray(kind[:, :10]) != 0).astype(np.float32)
    np.save(os.path.join(ds, "w.npy"), w); np.save(os.path.join(ds, "uw.npy"), uw)
    r_old = np.array([g["reward"] for g in games]); r_new = np.array([g["reward2"] for g in games])
    print(json.dumps({"rows": int(len(gid)), "games": len(games), "beta": round(beta, 1), "ess_frac": round(float(ess), 3),
                      "adv_sd": round(float(A.std()), 1), "reward_old_mean": round(float(r_old.mean()), 1),
                      "reward_new_mean": round(float(r_new.mean()), 1), "corr_old_new": round(float(np.corrcoef(r_old, r_new)[0, 1]), 3),
                      "w_max": round(float(gw.max()), 2)}), flush=True)


if __name__ == "__main__":
    main()
