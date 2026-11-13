"""evo_loop -- autonomous (1+lambda) hill climb over v53/no-camp PARAMS.

Pre-registered protocol (written before the first run):
  * Champion starts as nocamp (v53 + soft_w 0, campers 1, camp_from 12).
  * Round r: LAMBDA single-parameter mutations of the champion; each is
    screened with SCREEN seeds x both sides vs the champion, all candidates
    on the SAME seeds (common random numbers). Seed blocks never repeat.
  * Best screened candidate with mean > SCREEN_BAR goes to confirmation:
    CONFIRM fresh seeds x 2 vs champion + SANITY seeds x 2 vs base_v53.
  * Promote iff confirm mean > 0 AND pooled (screen+confirm) mean > 2 se
    AND sanity mean vs base_v53 > SANITY_BAR AND no errors AND worst step
    < 400 ms. Otherwise the mutation is recorded as rejected.
State: ../data/evo_state.json (resumable). Every game: ../data/evo_results.jsonl.

Usage: python3 evo_loop.py [minutes]
"""

import json
import os
import random
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evo import make_variant, run_pairs, summarize  # noqa: E402

STATE = os.path.join(HERE, "..", "data", "evo_state.json")
PARENT = "nocamp.py"
BASE = "base_v53.py"
LAMBDA = 4
SCREEN = 6        # seeds -> 12 games per candidate
CONFIRM = 16      # seeds -> 32 games
SANITY = 4        # seeds -> 8 games vs base_v53
SCREEN_BAR = 400.0
SANITY_BAR = 2000.0
SEED0 = 100000    # evo seed space, disjoint from all earlier evals

# mutation space: name -> (kind, lo, hi)
SPACE = {
    "H": ("int", 8, 18), "S": ("int", 3, 8), "gamma": ("f", 0.8, 0.98),
    "rho": ("f", 0.0, 0.12), "rho_early": ("f", 0.0, 0.2), "mu": ("f", 0.0, 0.3),
    "mu_early": ("f", 0.0, 0.8), "term_k": ("f", 0.3, 2.0),
    "tie_cost": ("f", 0.0, 40.0), "kill_c": ("f", 0.0, 120.0), "kill_p": ("f", 0.0, 120.0),
    "info_c": ("f", 0.0, 8.0), "info_p": ("f", 0.0, 10.0),
    "planters": ("int", 0, 3), "planters_late": ("int", 0, 3), "planter_from": ("int", 2, 8),
    "plant_ratio": ("f", 1.0, 3.0), "plant_min": ("f", 0.0, 40.0), "hold_p": ("f", 0.4, 1.0),
    "steal_w": ("f", 0.0, 1.5), "cell_loss_w": ("f", 0.0, 1.5), "seize_min": ("f", 0.0, 30.0),
    "pdisc": ("f", 0.6, 0.98), "pp_cost": ("f", 0.0, 30.0), "pp_stand": ("f", 5.0, 60.0),
    "guard_w": ("f", 0.0, 1.5), "guard_min": ("f", 0.0, 60.0),
    "recruit_until": ("int", 15, 80), "endgame": ("int", 6, 30),
    "hunt_min": ("f", 50.0, 400.0), "hunt_R": ("int", 2, 9), "hunt_n": ("int", 1, 5),
    "hunt_targets": ("int", 1, 5), "hunt_cool": ("int", 3, 40),
    "shadows": ("int", 0, 4), "shadow_pen": ("f", 0.0, 10.0),
    "run_small": ("f", 40.0, 300.0), "run_b1": ("f", 50.0, 400.0), "run_b2": ("f", 100.0, 800.0),
    "carry_ok": ("f", 30.0, 300.0), "gate_min": ("f", 20.0, 150.0), "gate_w": ("f", 0.0, 1.5),
    "retire_from": ("int", 40, 200), "retire_stock": ("f", 1000.0, 5000.0),
    "early_planters": ("int", 0, 4), "avoid1": ("f", 0.0, 1.5), "avoid2": ("f", 0.0, 0.6),
    "sweep_w": ("f", 0.0, 1.0), "base_def": ("f", 0.0, 60.0), "camp_from": ("int", 6, 40),
    "campers": ("int", 0, 3), "camper_max": ("f", 5.0, 60.0),
}


def base_params():
    import importlib.util
    spec = importlib.util.spec_from_file_location("evo_parent", os.path.join(HERE, PARENT))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return dict(m.PARAMS)


def mutate(cur, rng, rejected):
    for _ in range(200):
        k = rng.choice(sorted(SPACE))
        kind, lo, hi = SPACE[k]
        v = cur[k]
        if kind == "int":
            step = max(1, int(round(abs(v) * 0.25)))
            nv = v + rng.choice([-1, 1]) * step
            nv = int(min(hi, max(lo, nv)))
        else:
            if v == 0:
                nv = rng.uniform(lo, lo + (hi - lo) * 0.25)
            else:
                nv = v * rng.choice([0.6, 0.8, 1.25, 1.6])
            nv = round(min(hi, max(lo, nv)), 4)
        if nv != v and f"{k}={nv}" not in rejected:
            return k, nv
    return None


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"champion": {}, "champ_id": "c0", "round": 0, "next_seed": SEED0,
            "rejected": [], "promotions": [], "rounds": []}


def save_state(s):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, STATE)


def pooled(rows):
    m = [r["margin"] for r in rows if "margin" in r]
    return st.mean(m), st.stdev(m) / len(m) ** 0.5


def main(minutes):
    deadline = time.time() + 60 * minutes
    s = load_state()
    bp = base_params()
    rng = random.Random(12345 + s["round"])
    while time.time() < deadline:
        s["round"] += 1
        r = s["round"]
        champ = dict(s["champion"])
        cur = dict(bp); cur.update(champ)
        champ_file = make_variant(PARENT, champ, s["champ_id"])
        cands = []
        seen = set()
        while len(cands) < LAMBDA:
            mu = mutate(cur, rng, set(s["rejected"]))
            if mu is None or mu[0] in seen:
                continue
            seen.add(mu[0])
            ov = dict(champ); ov[mu[0]] = mu[1]
            cands.append((mu, make_variant(PARENT, ov, f"r{r}_{mu[0]}")))
        sd = s["next_seed"]; s["next_seed"] += SCREEN
        rows = run_pairs([(f, champ_file, sd, SCREEN) for _, f in cands], tag=f"evo_r{r}_screen")
        res = []
        for mu, f in cands:
            rr = [x for x in rows if x.get("a_file") == f]
            res.append((summarize(rr), mu, f, rr))
        res.sort(key=lambda x: -(x[0].get("mean", -1e9) if x[0]["err"] == 0 else -1e9))
        rec = {"round": r, "champ": s["champ_id"], "screen": [
            {"mut": f"{mu[0]}={mu[1]}", **sm} for sm, mu, _, _ in res]}
        best_sm, best_mu, best_f, best_rows = res[0]
        verdict = "no_candidate"
        if best_sm["err"] == 0 and best_sm.get("mean", -1e9) > SCREEN_BAR:
            sd2 = s["next_seed"]; s["next_seed"] += CONFIRM
            sd3 = s["next_seed"]; s["next_seed"] += SANITY
            crow = run_pairs([(best_f, champ_file, sd2, CONFIRM), (best_f, BASE, sd3, SANITY)],
                             tag=f"evo_r{r}_confirm")
            conf = [x for x in crow if x.get("b_file") == champ_file]
            san = [x for x in crow if x.get("b_file") == BASE]
            cs, ss = summarize(conf), summarize(san)
            pm, pse = pooled(best_rows + conf)
            rec.update({"confirm": cs, "sanity": ss, "pooled": [round(pm, 1), round(pse, 1)]})
            ok = (cs["err"] == 0 and ss["err"] == 0 and cs["mean"] > 0 and pm > 2 * pse
                  and ss["mean"] > SANITY_BAR and max(cs["worst_ms"], ss["worst_ms"]) < 400)
            if ok:
                new_id = f"c{len(s['promotions']) + 1}"
                s["champion"][best_mu[0]] = best_mu[1]
                s["promotions"].append({"round": r, "id": new_id, "mut": f"{best_mu[0]}={best_mu[1]}",
                                        "pooled": [round(pm, 1), round(pse, 1)], "sanity": ss["mean"]})
                s["champ_id"] = new_id
                verdict = "PROMOTED " + new_id
            else:
                verdict = "rejected_confirm"
                s["rejected"].append(f"{best_mu[0]}={best_mu[1]}")
        for sm, mu, _, _ in res[1:]:
            if sm.get("mean", -1e9) < -SCREEN_BAR:
                s["rejected"].append(f"{mu[0]}={mu[1]}")
        rec["verdict"] = verdict
        s["rounds"].append(rec)
        save_state(s)
        print(json.dumps({"round": r, "verdict": verdict, "best": rec["screen"][0],
                          "champion": s["champion"]}), flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 100)
