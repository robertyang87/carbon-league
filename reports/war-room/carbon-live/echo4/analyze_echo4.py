#!/usr/bin/env python3
"""Echo4 launch exam analysis (C1, 2026-11-15).
F1 e4pack_m2: pack 819250bf vs Master2 wrapper 2aebfbc1, seeds 996000-996199
F2 e4bare_m2: bare f4fff163 vs Master2 wrapper 2aebfbc1, seeds 996000-996199
F3 e4pack_e2: pack 819250bf vs Echo2 15b6d754, seeds 999000-999199
Margins/U are A-side (first-named) perspective. U = 0.5*W + 0.5*sigmoid(m/1000).
Equivalence: paired-by-seed margin diff (pack - bare) with cluster bootstrap CI.
Calibration: score = F1 mean margin - 785; gate +150.
"""
import json, math, random, statistics as st

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29/"
HH = "354bcd1bc2dc8bc188c813f53d28c893"

def load(tag):
    rows = [json.loads(l) for l in open(BASE + tag + ".jsonl")]
    return rows

def u_of(margin):
    w = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-margin / 1000.0))

def stats(rows, ah, bh, seed0):
    assert len(rows) == 200, len(rows)
    errs = sum(1 for r in rows if r["err"])
    assert errs == 0, errs
    for r in rows:
        assert r["ah"] == ah and r["bh"] == bh and r["hh"] == HH, (r["ah"], r["bh"], r["hh"])
    seeds = sorted(r["seed"] for r in rows)
    assert seeds == list(range(seed0, seed0 + 200)), (seeds[0], seeds[-1])
    ms = [r["margin"] for r in rows]
    us = [u_of(m) for m in ms]
    dev = max(abs(r["u"] - u) for r, u in zip(rows, us))
    wins = sum(1 for m in ms if m > 0); draws = sum(1 for m in ms if m == 0)
    mean = st.mean(ms); se = st.stdev(ms) / math.sqrt(len(ms))
    mu = st.mean(us)
    by_seed = {r["seed"]: r["margin"] for r in rows}
    return {"wins": wins, "draws": draws, "losses": 200 - wins - draws,
            "mean_margin": mean, "se_margin": se, "mean_u": mu,
            "du": 2 * mu - 1, "u_dev_max": dev, "by_seed": by_seed}

f1 = stats(load("e4pack_m2"), "819250bfc82632786063dabd46a616bd", "2aebfbc1008d3475ece8e178696e292d", 996000)
f2 = stats(load("e4bare_m2"), "f4fff1638e51efd9e2c07720831627a4", "2aebfbc1008d3475ece8e178696e292d", 996000)
f3 = stats(load("e4pack_e2"), "819250bfc82632786063dabd46a616bd", "15b6d754e6e21de33870074d3983a16f", 999000)

# Paired equivalence diff by seed (pack - bare), cluster bootstrap CI95
diffs = [f1["by_seed"][s] - f2["by_seed"][s] for s in range(996000, 996200)]
dmean = st.mean(diffs); dse = st.stdev(diffs) / math.sqrt(len(diffs))
rng = random.Random(20261115)
boots = []
for _ in range(20000):
    boots.append(st.mean(rng.choice(diffs) for _ in range(200)))
boots.sort()
ci = (boots[500], boots[19500])
cal = f1["mean_margin"] - 785.0

out = {
    "F1_pack_vs_master2": {k: f1[k] for k in ("wins", "draws", "losses", "mean_margin", "se_margin", "mean_u", "du", "u_dev_max")},
    "F2_bare_vs_master2": {k: f2[k] for k in ("wins", "draws", "losses", "mean_margin", "se_margin", "mean_u", "du", "u_dev_max")},
    "F3_pack_vs_echo2": {k: f3[k] for k in ("wins", "draws", "losses", "mean_margin", "se_margin", "mean_u", "du", "u_dev_max")},
    "equivalence_paired_diff_pack_minus_bare": {"mean": dmean, "se": dse, "ci95_boot": ci,
                                                "within_noise_band_ci_includes_0": ci[0] <= 0 <= ci[1]},
    "calibration": {"score": cal, "gate": 150.0, "pass": cal >= 150.0},
}
print(json.dumps(out, indent=1))
with open("/home/hatch/workspace/carbon-live/echo4/echo4-analysis.json", "w") as f:
    json.dump(out, f, indent=1)
