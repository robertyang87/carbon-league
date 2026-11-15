#!/usr/bin/env python3
"""Merge Stage-1 endgame pilot sub-runs A+B (runs ON Studio).
Replicates the gate's cluster_stats / cluster_bootstrap_ci definitions
exactly (per-game means, t CI over games, cluster bootstrap seed 996100).
Padding roots (teacher_calls==0 or du==0) are counted separately and
EXCLUDED from all inference. Writes endgame_p1_merged.json."""
import json, math, os, sys
import numpy as np
from scipy import stats

P1 = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/p1ab-20261114"

def t_ci(vals):
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    crit = float(stats.t.ppf(0.975, n - 1))
    half = crit * sd / math.sqrt(n)
    return m, m - half, m + half

def cluster_stats(rows):
    by_game = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(float(r["du"]))
    gm = np.array([np.mean(v) for v in by_game.values()])
    m, lo, hi = t_ci(gm)
    rv = np.array([float(r["du"]) for r in rows])
    return {"n_rows": len(rows), "n_games": len(by_game),
            "root_mean": float(rv.mean()),
            "game_mean": m, "game_ci95_t": [lo, hi],
            "sigma_game": float(gm.std(ddof=1)) if len(gm) > 1 else None,
            "sigma_root": float(rv.std(ddof=1)) if len(rv) > 1 else None}

def cluster_boot(rows, seed=996100, n_boot=10000):
    by_game = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(float(r["du"]))
    games = list(by_game)
    gm = np.array([np.mean(by_game[g]) for g in games])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(games), size=(n_boot, len(games)))
    means = gm[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

def tertile(rows, lo, hi):
    sel = [r for r in rows if lo <= r["step"] <= hi]
    if not sel:
        return {"n_rows": 0, "n_games": 0, "game_mean": None}
    s = cluster_stats(sel)
    return {"n_rows": s["n_rows"], "n_games": s["n_games"],
            "game_mean": s["game_mean"], "sigma_game": s["sigma_game"],
            "game_ci95_t": s["game_ci95_t"]}

res = {}
rows_all, padding = [], []
ftmap = {}
_ft = json.load(open(os.path.join(P1, "first_terminal_shard0.json")))
for g, v in _ft.items():
    ftmap[g] = v["ft"]
for tag in ("a", "b"):
    d = json.load(open(os.path.join(P1, "endgame_p1%s.json" % tag)))
    sc = json.load(open(os.path.join(P1, "endgame_p1%s.json.pilot.json" % tag)))
    rows = d["rows"]
    res["subrun_%s" % tag] = {
        "n_rows": len(rows), "wall_s": d["wall_s"],
        "artifact_id": sc["artifact_id"], "pilot_run_id": sc["pilot_run_id"],
        "grade": sc["grade"], "sigma_game": sc["sigma_game"],
        "sigma_root": sc["sigma_root"],
        "du_gate": d["du_gate"],
    }
    for r in rows:
        (padding if (r["teacher_calls"] == 0 or r["du"] == 0.0) else rows_all).append(
            dict(r, subrun=tag))
    res["subrun_%s" % tag]["padding_n"] = sum(
        1 for r in rows if r["teacher_calls"] == 0 or r["du"] == 0.0)

# Join against per-game first-terminal steps (k0_aux_labels definition):
# a row is join-padding iff step >= first_terminal(game). Cross-tab with
# the zero-row signature (teacher_calls==0 or du==0).
allrows = []
for tag in ("a", "b"):
    d = json.load(open(os.path.join(P1, "endgame_p1%s.json" % tag)))
    allrows += [dict(r, subrun=tag) for r in d["rows"]]
join = {"step_ge_ft": 0, "step_ge_ft_and_zero": 0, "step_ge_ft_but_nonzero": [],
        "step_lt_ft_but_zero": [], "step_lt_ft": 0}
for r in allrows:
    zero = (r["teacher_calls"] == 0 or r["du"] == 0.0)
    if r["step"] >= ftmap[r["game"]]:
        join["step_ge_ft"] += 1
        if zero:
            join["step_ge_ft_and_zero"] += 1
        else:
            join["step_ge_ft_but_nonzero"].append(
                {"game": r["game"], "step": r["step"], "subrun": r["subrun"]})
    else:
        join["step_lt_ft"] += 1
        if zero:
            join["step_lt_ft_but_zero"].append(
                {"game": r["game"], "step": r["step"], "subrun": r["subrun"],
                 "teacher_calls": r["teacher_calls"], "du": r["du"]})
res["join_vs_first_terminal"] = join
join_real = [r for r in allrows if r["step"] < ftmap[r["game"]]]
res["join_real_stats"] = cluster_stats(join_real)
res["join_real_stats"]["game_ci95_boot_cluster"] = cluster_boot(join_real)

res["padding_rows"] = [{"game": r["game"], "step": r["step"], "subrun": r["subrun"],
                        "teacher_calls": r["teacher_calls"], "du": r["du"]}
                       for r in padding]
res["n_padding"] = len(padding)
res["n_real_rows"] = len(rows_all)
res["n_real_games"] = len(set(r["game"] for r in rows_all))
res["per_game_root_counts"] = {}
for r in rows_all:
    res["per_game_root_counts"][r["game"]] = res["per_game_root_counts"].get(r["game"], 0) + 1
from collections import Counter
res["root_count_distribution"] = dict(Counter(res["per_game_root_counts"].values()))
merged = cluster_stats(rows_all)
merged["game_ci95_boot_cluster"] = cluster_boot(rows_all)
res["merged"] = merged
res["tertiles"] = {
    "200-229": tertile(rows_all, 200, 229),
    "230-259": tertile(rows_all, 230, 259),
    "260-299": tertile(rows_all, 260, 299),
}
res["teacher_totals"] = {
    "calls": sum(r["teacher_calls"] for r in rows_all),
    "ms": sum(r["teacher_ms"] for r in rows_all),
}
res["wall_total_s"] = res["subrun_a"]["wall_s"] + res["subrun_b"]["wall_s"]
json.dump(res, open(os.path.join(P1, "endgame_p1_merged.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
