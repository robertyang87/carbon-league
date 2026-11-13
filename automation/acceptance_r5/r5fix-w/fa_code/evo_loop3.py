"""evo_loop3 -- evo_loop2 with the corrected (non-inert) tree-heavy pool spar_tf2/spar_tf3. See evo_loop2 docstring for the protocol.

Why v2: v1 screened only vs the champion (self-play mirror). The replay
a2_highscore_win_v53sw0 shows no-camp loses to AlphaCarbon-2 through tree
economy (37 vs 17 plants) and slow banking (cargo taxed by enemy trees),
which pure mirror play cannot expose. v2 screens against a pool:
  champion (mirror), spar_tf (tree-heavy whole game), spar_tw_nc
  (tree-war schedule in steps 50-100), all on no-camp.

Pre-registered protocol:
  * Per round: the champion and each candidate play every pool opponent
    on the same SCREEN seeds x both sides (common random numbers).
    delta_o = mean margin(cand vs o) - mean margin(champ vs o); the
    mirror opponent's champion baseline is 0 by symmetry.
    score = mean over pool of delta_o.
  * Best candidate with score > SCREEN_BAR and no pool delta below
    -WORST_BAR goes to confirmation on fresh seeds (CONFIRM seeds per
    opponent, champion re-run on the same seeds) + SANITY vs base_v53.
  * Promote iff pooled (screen+confirm) score > 2 se, confirm score > 0,
    every confirm delta > -WORST_BAR, sanity vs base_v53 > SANITY_BAR,
    no errors, worst step < 400 ms.
  * Candidates come from QUEUE (diagnosis-driven single-variable
    hypotheses) first, then random single-parameter mutations.
State: ../data/evo3_state.json; games: ../data/evo_results.jsonl.
"""

import json
import os
import random
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evo import make_variant, run_pairs  # noqa: E402
from evo_loop import SPACE, mutate, base_params  # noqa: E402

STATE = os.path.join(HERE, "..", "data", "evo3_state.json")
PARENT = "nocamp.py"
BASE = "base_v53.py"
SPARS = ["exp/spar_tf2.py", "exp/spar_tf3.py"]
LAMBDA = 3
SCREEN = 4
CONFIRM = 8
SANITY = 3
SCREEN_BAR = 250.0
WORST_BAR = 800.0
SANITY_BAR = 2000.0
SEED0 = 300000

QUEUE = [
    {"H": 16}, {"planters": 2}, {"early_planters": 3}, {"plant_min": 6.0},
    {"plant_ratio": 1.3}, {"seize_min": 4.0}, {"endgame": 20}, {"endgame": 10},
    {"rho_early": 0.1}, {"recruit_until": 25}, {"recruit_until": 60},
    {"retire_from": 150}, {"steal_w": 0.9}, {"gamma": 0.95}, {"H": 9},
    {"mu": 0.1}, {"tie_cost": 25.0}, {"kill_c": 70.0}, {"hold_p": 0.95},
    {"pdisc": 0.92}, {"guard_min": 10.0}, {"planters_late": 2},
]


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"champion": {}, "champ_id": "c0", "round": 0, "next_seed": SEED0,
            "qpos": 0, "rejected": [], "promotions": [], "rounds": []}


def save_state(s):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, STATE)


def margins(rows, a, b):
    return [r["margin"] for r in rows if r.get("a_file") == a and r.get("b_file") == b and "margin" in r]


def errs(rows, a):
    return sum(1 for r in rows if r.get("a_file") == a and "error" in r)


def worst(rows, a):
    return max([r.get("a_worst_step_ms", 0) for r in rows if r.get("a_file") == a] or [0])


def evaluate(cands, champ_file, seed, n, tag):
    """Returns {cand_file: (score, per_opp_deltas, per_game_paired_list)}."""
    jobs = [(champ_file, o, seed, n) for o in SPARS]
    for f in cands:
        jobs += [(f, champ_file, seed, n)] + [(f, o, seed, n) for o in SPARS]
    rows = run_pairs(jobs, tag=tag)
    out = {}
    for f in cands:
        deltas, per_game = {}, []
        m = margins(rows, f, champ_file)
        deltas["mirror"] = st.mean(m) if m else float("nan")
        per_game += m
        for o in SPARS:
            mc, mb = margins(rows, f, o), margins(rows, champ_file, o)
            deltas[o] = (st.mean(mc) - st.mean(mb)) if mc and mb else float("nan")
            # paired per (seed, side): cand - champ
            per_game += [x - y for x, y in zip(mc, mb)]
        score = st.mean(deltas.values())
        out[f] = {"score": round(score, 1), "deltas": {k: round(v, 1) for k, v in deltas.items()},
                  "err": errs(rows, f) + errs(rows, champ_file), "worst_ms": worst(rows, f),
                  "per_game": per_game}
    return out


def main(minutes):
    deadline = time.time() + 60 * minutes
    s = load_state()
    bp = base_params()
    rng = random.Random(777 + s["round"])
    while time.time() < deadline:
        s["round"] += 1
        r = s["round"]
        champ = dict(s["champion"])
        cur = dict(bp); cur.update(champ)
        champ_file = make_variant(PARENT, champ, s["champ_id"])
        muts = []
        while len(muts) < LAMBDA and s["qpos"] < len(QUEUE):
            q = QUEUE[s["qpos"]]; s["qpos"] += 1
            k, v = next(iter(q.items()))
            if cur.get(k) != v and f"{k}={v}" not in s["rejected"]:
                muts.append((k, v))
        keys = {k for k, _ in muts}
        while len(muts) < LAMBDA:
            mu = mutate(cur, rng, set(s["rejected"]))
            if mu and mu[0] not in keys:
                keys.add(mu[0]); muts.append(mu)
        cands = {}
        for k, v in muts:
            ov = dict(champ); ov[k] = v
            cands[make_variant(PARENT, ov, f"e3r{r}_{k}")] = (k, v)
        sd = s["next_seed"]; s["next_seed"] += SCREEN
        res = evaluate(list(cands), champ_file, sd, SCREEN, f"evo3_r{r}_screen")
        ranked = sorted(res.items(), key=lambda kv: -(kv[1]["score"] if kv[1]["err"] == 0 else -1e9))
        rec = {"round": r, "champ": s["champ_id"],
               "screen": [{"mut": "%s=%s" % cands[f], **{k: v for k, v in x.items() if k != "per_game"}}
                          for f, x in ranked]}
        bf, bx = ranked[0]
        bk, bv = cands[bf]
        verdict = "no_candidate"
        if (bx["err"] == 0 and bx["score"] > SCREEN_BAR
                and min(bx["deltas"].values()) > -WORST_BAR):
            sd2 = s["next_seed"]; s["next_seed"] += CONFIRM
            cres = evaluate([bf], champ_file, sd2, CONFIRM, f"evo3_r{r}_confirm")[bf]
            sd3 = s["next_seed"]; s["next_seed"] += SANITY
            srows = run_pairs([(bf, BASE, sd3, SANITY)], tag=f"evo3_r{r}_sanity")
            sm = margins(srows, bf, BASE)
            san = st.mean(sm) if sm else -1e9
            pg = bx["per_game"] + cres["per_game"]
            pm, pse = st.mean(pg), st.stdev(pg) / len(pg) ** 0.5
            rec.update({"confirm": {k: v for k, v in cres.items() if k != "per_game"},
                        "sanity": round(san, 1), "pooled": [round(pm, 1), round(pse, 1)]})
            ok = (cres["err"] == 0 and cres["score"] > 0 and pm > 2 * pse
                  and min(cres["deltas"].values()) > -WORST_BAR and san > SANITY_BAR
                  and max(cres["worst_ms"], bx["worst_ms"]) < 400
                  and errs(srows, bf) == 0)
            if ok:
                nid = f"c{len(s['promotions']) + 1}"
                s["champion"][bk] = bv
                s["promotions"].append({"round": r, "id": nid, "mut": f"{bk}={bv}",
                                        "pooled": [round(pm, 1), round(pse, 1)], "sanity": round(san, 1)})
                s["champ_id"] = nid
                verdict = "PROMOTED " + nid
            else:
                verdict = "rejected_confirm"
                s["rejected"].append(f"{bk}={bv}")
        for f, x in ranked[1:]:
            if x["score"] < -SCREEN_BAR:
                s["rejected"].append("%s=%s" % cands[f])
        rec["verdict"] = verdict
        s["rounds"].append(rec)
        save_state(s)
        print(json.dumps({"round": r, "verdict": verdict, "screen": rec["screen"],
                          "confirm": rec.get("confirm"), "pooled": rec.get("pooled"),
                          "champion": s["champion"]}), flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 100)
