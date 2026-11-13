"""analyze_s3p2: verdict computations for the S3 v2 batch.

Reads eval_s3p2_results.jsonl (+ v1 eval_s3_results.jsonl for the
honly cross-check and v1 stranded reference). Prints and dumps
s3p2_analysis.json:
  * sanity: off2 == base-vs-base margins; honly2 == v1 honly margins
  * phase net effect per (seed, side) = both2 - honly2, per-seed
    sums, mean; promotion test (>=3/4 seeds positive AND mean > +300)
  * stranded distributions (fork side) both2 vs honly2, bucketed
  * deep-behind (entry250 <= -1200) per-game banking table
  * recruit-gate accounting: authorised / observed / paid-back,
    widened vs v1 numbers, norec ablation delta
  * timing
"""

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def load(fn):
    out = {}
    with open(os.path.join(HERE, fn)) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[(r["tag"], r["seed"], r["policy_side"])] = r
    return out


R2 = load("eval_s3p2_results.jsonl")
R1 = load("eval_s3_results.jsonl")
A = {}

# ---- sanity ------------------------------------------------------------
off_ok = True
for side in (0, 1):
    a = R2[("s3p2off", 1000, side)]["margin"]
    b = R2[("basebase", 1000, side)]["margin"]
    if a != b:
        off_ok = False
        print(f"SANITY FAIL off2 seed1000 side{side}: {a} vs basebase {b}")
A["sanity_off_matches_basebase"] = off_ok

hon_ok = True
for s in (1000, 1001, 1002, 1003):
    for side in (0, 1):
        a = R2[("s3p2honly", s, side)]["margin"]
        b = R1[("s3honly", s, side)]["margin"]
        if a != b:
            hon_ok = False
            print(f"HONLY XCHECK FAIL seed{s} side{side}: v2 {a} vs v1 {b}")
A["sanity_honly2_matches_v1_honly"] = hon_ok

# ---- phase net effect ---------------------------------------------------
per_game = []
per_seed = {}
for s in (1000, 1001, 1002, 1003):
    tot = 0.0
    for side in (0, 1):
        mb = R2[("s3p2both", s, side)]["margin"]
        mh = R2[("s3p2honly", s, side)]["margin"]
        eff = round(mb - mh, 2)
        per_game.append({"seed": s, "side": side, "both": mb,
                         "honly": mh, "phase_effect": eff,
                         "entry250": R2[("s3p2both", s, side)]["margin_entry250"]})
        tot += eff
    per_seed[s] = round(tot, 2)
mean_eff = round(statistics.mean(g["phase_effect"] for g in per_game), 2)
pos_seeds = sum(1 for v in per_seed.values() if v > 0)
A["phase_net_effect"] = {
    "per_game": per_game, "per_seed": per_seed, "mean": mean_eff,
    "positive_seeds": pos_seeds,
    "promotion_test": bool(pos_seeds >= 3 and mean_eff > 300),
}
raw_both = [R2[("s3p2both", s, side)]["margin"]
            for s in (1000, 1001, 1002, 1003) for side in (0, 1)]
A["both2_raw_vs_base"] = {"margins": raw_both,
                          "mean": round(statistics.mean(raw_both), 2),
                          "wins": sum(1 for m in raw_both if m > 0)}

# ---- stranded distribution ----------------------------------------------
def stranded(tag):
    return [R2[(tag, s, side)]["ledger_pol"]["stranded"]
            for s in (1000, 1001, 1002, 1003) for side in (0, 1)]

def buckets(v):
    return {"le200": sum(1 for x in v if x <= 200),
            "mid": sum(1 for x in v if 200 < x < 800),
            "ge800": sum(1 for x in v if x >= 800)}

sb, sh = stranded("s3p2both"), stranded("s3p2honly")
A["stranded"] = {
    "both2": sorted(sb), "honly2": sorted(sh),
    "both2_mean": round(statistics.mean(sb), 1),
    "honly2_mean": round(statistics.mean(sh), 1),
    "both2_worst": max(sb), "honly2_worst": max(sh),
    "both2_buckets": buckets(sb), "honly2_buckets": buckets(sh),
    "v1_both_reference": [43, 61, 100, 100, 857, 1289, 1341, 1529],
}

# ---- ledgers --------------------------------------------------------------
def ledger_mean(tag, key):
    return round(statistics.mean(
        R2[(tag, s, side)]["ledger_pol"][key]
        for s in (1000, 1001, 1002, 1003) for side in (0, 1)), 1)

A["ledger_both2"] = {k: ledger_mean("s3p2both", k) for k in
                     ("deposits", "tree_income", "death_cargo", "deaths",
                      "stranded")}
A["ledger_honly2"] = {k: ledger_mean("s3p2honly", k) for k in
                      ("deposits", "tree_income", "death_cargo", "deaths",
                       "stranded")}
A["dep_last50"] = {
    "both2_pol": round(statistics.mean(
        R2[("s3p2both", s, side)]["dep_last50_pol"]
        for s in (1000, 1001, 1002, 1003) for side in (0, 1)), 1),
    "honly2_pol": round(statistics.mean(
        R2[("s3p2honly", s, side)]["dep_last50_pol"]
        for s in (1000, 1001, 1002, 1003) for side in (0, 1)), 1),
}

# ---- deep-behind / banking table -----------------------------------------
deep = []
for g in per_game:
    if g["entry250"] <= -1200:
        r = R2[("s3p2both", g["seed"], g["side"])]
        deep.append({
            "seed": g["seed"], "side": g["side"], "entry250": g["entry250"],
            "bank_steps": r["bot_pol"].get("s3p2", {}).get("steps_bank"),
            "gamble_steps": r["bot_pol"].get("s3p2", {}).get("steps_gamble"),
            "transitions": r["bot_pol"].get("s3p2", {}).get("transitions"),
            "final_both2": g["both"], "final_honly2": g["honly"],
            "chase_both2": round(g["both"] - g["entry250"], 2),
            "chase_honly2": round(g["honly"] - g["entry250"], 2),
            "phase_effect": g["phase_effect"],
        })
A["deep_behind_games"] = deep

# mode occupancy across both2 games
occ = {"steps_bank": 0, "steps_protect": 0, "steps_gamble": 0,
       "steps_neutral": 0, "transitions": 0}
for s in (1000, 1001, 1002, 1003):
    for side in (0, 1):
        st = R2[("s3p2both", s, side)]["bot_pol"].get("s3p2", {})
        for k in occ:
            occ[k] += st.get(k, 0)
A["mode_occupancy_both2"] = occ

# ---- recruit gate ---------------------------------------------------------
def rec_agg(rows, who):
    n = pb = 0
    deps = []
    for r in rows:
        rr = r[who]
        n += rr["n"]
        pb += rr["paid_back"]
        deps.extend(rr["deps"])
    return {"n": n, "paid_back": pb, "deps": sorted(deps)}

both_rows = [R2[("s3p2both", s, side)]
             for s in (1000, 1001, 1002, 1003) for side in (0, 1)]
honly_rows = [R2[("s3p2honly", s, side)]
              for s in (1000, 1001, 1002, 1003) for side in (0, 1)]
auth = sum(r["bot_pol"].get("s3", {}).get("phase_recruits", 0)
           for r in both_rows)
A["recruit_gate"] = {
    "authorised_both2": auth,
    "observed_pol_both2": rec_agg(both_rows, "recruits_pol"),
    "background_both2_opp": rec_agg(both_rows, "recruits_opp"),
    "background_honly2_pol": rec_agg(honly_rows, "recruits_pol"),
    "background_honly2_opp": rec_agg(honly_rows, "recruits_opp"),
    "v1_reference": {"authorised": 3, "games_triggered": 1,
                     "observed_deps_window": [0, 328]},
}
norec_eff = []
for s in (1000, 1001):
    for side in (0, 1):
        norec_eff.append(round(
            R2[("s3p2both", s, side)]["margin"]
            - R2[("s3p2norec", s, side)]["margin"], 2))
A["recruit_gate"]["norec_ablation_both2_minus_norec"] = {
    "per_game": norec_eff,
    "mean": round(statistics.mean(norec_eff), 2)}

# ---- timing ----------------------------------------------------------------
ms = [R2[("s3p2both", s, side)]["a_ms_per_step"]
      for s in (1000, 1001, 1002, 1003) for side in (0, 1)]
worst = max(R2[("s3p2both", s, side)]["a_worst_step_ms"]
            for s in (1000, 1001, 1002, 1003) for side in (0, 1))
A["timing_both2_ms_per_step"] = {"mean": round(statistics.mean(ms), 2),
                                 "max_game_mean": max(ms),
                                 "worst_single_step": worst}

with open(os.path.join(HERE, "s3p2_analysis.json"), "w") as f:
    json.dump(A, f, indent=1, ensure_ascii=False)
print(json.dumps(A, indent=1, ensure_ascii=False))
