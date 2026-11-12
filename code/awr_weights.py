"""awr_weights DS CORPUS_DIR [CORPUS_DIR ...] -- AWR weights on a build_awr dataset.

Reward (recomputed from each replay's final state at step 100, our side minus theirs):
  cash + cargo + sum over standing trees of  4-neighbour carbon * (1 - 0.9625^min(50 - age, 30)) * p_survive,
  p_survive = 0.3 if an enemy planter stands within 1 step of the tree (it is about to be seized), else 0.8
  (2026-11-09 review: a reward taken at the step-100 instant overvalues trees planted for the opponent to take).
Whole games (300 steps in the replay, awr_gen with AWR_STEPS=300): the reward is the final cash difference instead.
Advantage: leave-one-out baseline over all games of the same seed (the greedy game and the K sampled ones):
  A_i = R_i - mean(R_j, j != i)   (lower variance than "minus the greedy game" alone).
Weights: w = exp(A / beta); beta starts at std(A) and grows until the effective sample size
  (sum w)^2 / sum w^2 is at least ESS_MIN (default 20%) of the games -- otherwise a few lucky games dominate.
uw (per unit) = 1 for planters and base slots, 0 for our collectors (cars stay the net's own).
AWR_CAR_ANCHOR=c (default 0): car tokens get the constant weight c regardless of the game's weight -- a self-anchor that
keeps the cars where they are when no separate anchor dataset is mixed in (use 1)."""
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
        g["whole"] = len(st) >= 300 or bool(os.environ.get("AWR_FINAL"))
        if len(st) >= 300 or os.environ.get("AWR_FINAL"):      # whole game: the reward is the final score difference
            g["reward2"] = float(o["players"][g["side"]][0]) - float(o["players"][1 - g["side"]][0])
        else:
            g["reward2"] = side_value(o, carbon, g["side"]) - side_value(o, carbon, 1 - g["side"])
        g["corpus"] = os.path.basename(corpus.rstrip("/"))
    by = defaultdict(list)
    for g in games:
        # a 300-step game that ends exactly level is a perfect mirror game (our net against an identical deterministic
        # copy on the 4-fold symmetric map: both sides play the same moves, units cancel on the symmetry lines, some
        # games never develop at all). It carries no information: weight 0 and not part of the seed's baseline.
        g["mirror"] = bool(g.get("whole") and abs(g["reward2"]) < 0.05)       # same rule as awr_signal (reward rounded to 0.1)
        if not g["mirror"]:
            by[g["seed"]].append(g)
    for g in games:
        g["adv2"] = 0.0
    for gs in by.values():
        tot = sum(x["reward2"] for x in gs)
        for x in gs:
            x["adv2"] = x["reward2"] - (tot - x["reward2"]) / max(1, len(gs) - 1) if len(gs) > 1 else 0.0
    return games


def main():
    ds, corpora = sys.argv[1], sys.argv[2:]
    games = [g for c in corpora for g in rescore(c)]
    nmir = sum(g["mirror"] for g in games)
    A = np.array([g["adv2"] for g in games if not g["mirror"]])
    ess_min = float(os.environ.get("AWR_ESS_MIN", "0.2")); wmax = float(os.environ.get("AWR_WMAX", "20"))
    beta = float(A.std()) or 1.0
    for _ in range(40):
        w_ = np.minimum(wmax, np.exp(A / beta))
        ess = w_.sum() ** 2 / (w_ ** 2).sum() / len(w_)
        if ess >= ess_min:
            break
        beta *= 1.25
    wkey = {g["key"]: 0.0 if g["mirror"] else float(min(wmax, np.exp(g["adv2"] / beta))) for g in games}
    meta = json.load(open(os.path.join(ds, "meta.json"))); gid = np.load(os.path.join(ds, "gid.npy"))
    if len(wkey) != len(games):
        sys.exit(f"awr_weights: {len(games)} games but only {len(wkey)} distinct keys -- the corpora share game keys "
                 f"(same seed range). Run awr_rekey.py on each corpus (a different TAG each) and rebuild the dataset.")
    if len(meta["games"]) != len(games):
        sys.exit(f"awr_weights: the dataset holds {len(meta['games'])} games, the corpora {len(games)} -- rebuild the dataset "
                 f"from exactly these corpora (cnet2_build drops games whose file name it has already seen).")
    gw = np.array([wkey[g["key"].replace(".json.gz", "")] for g in meta["games"]], np.float32)
    w = gw[gid].astype(np.float32)
    kind = np.load(os.path.join(ds, "tok_kind.npy"), mmap_mode="r")
    S = np.load(os.path.join(ds, "y.npy"), mmap_mode="r").shape[1]
    uw = np.ones((len(gid), S), np.float32)
    uw[:, :10] = (np.asarray(kind[:, :10]) != 0).astype(np.float32)
    # Self-anchor for the cars. With weight 0 nothing holds the car outputs while the shared decoder is trained on the
    # planter / base rows (round sp1 without an anchor set: car agreement 99.2% -> 93.0%). The cars in these games are
    # never sampled, so their labels are the net's own moves: give every car token the constant weight AWR_CAR_ANCHOR,
    # whatever the game's weight (the trainer multiplies w * uw, hence the division).
    car_anchor = float(os.environ.get("AWR_CAR_ANCHOR", "0"))
    if car_anchor > 0:
        iscar = np.asarray(kind[:, :10]) == 0
        uw[:, :10] = np.where(iscar, car_anchor / np.maximum(w, 1e-6)[:, None], uw[:, :10])
    np.save(os.path.join(ds, "w.npy"), w); np.save(os.path.join(ds, "uw.npy"), uw)
    r_old = np.array([g["reward"] for g in games]); r_new = np.array([g["reward2"] for g in games])
    print(json.dumps({"rows": int(len(gid)), "games": len(games), "mirror_dropped": int(nmir), "beta": round(beta, 1), "ess_frac": round(float(ess), 3),
                      "adv_sd": round(float(A.std()), 1), "reward_old_mean": round(float(r_old.mean()), 1),
                      "reward_new_mean": round(float(r_new.mean()), 1), "corr_old_new": round(float(np.corrcoef(r_old, r_new)[0, 1]), 3),
                      "w_max": round(float(gw.max()), 2), "car_anchor": car_anchor}), flush=True)


if __name__ == "__main__":
    main()
