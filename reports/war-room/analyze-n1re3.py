import hashlib
import json
import math
import os
import random

SNAP = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"


def u_of(margin):
    w = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-margin / 1000.0))


raw = open(os.path.join(SNAP, "n1r_echo3.final.jsonl"), "rb").read()
rows = {}
for ln in raw.decode().splitlines():
    if ln.strip():
        r = json.loads(ln)
        rows[r["seed"]] = r
assert set(rows) == set(range(989000, 989200)), "seed set"
assert all(r["err"] is None for r in rows.values()), "err all None"
assert all(r["ah"] == "ba7f84c3652ea2cd14a2243f2b1eace2" for r in rows.values()), "A pack md5"
assert all(r["bh"] == "7b7620499411a088cf4dc17c0e929bd7" for r in rows.values()), "B pack md5"
assert all(r["hh"] == "354bcd1bc2dc8bc188c813f53d28c893" for r in rows.values()), "harness md5"
max_dev = max(abs(r["u"] - u_of(r["margin"])) for r in rows.values())

n = len(rows)
us = [rows[s]["u"] for s in sorted(rows)]
margins = [rows[s]["margin"] for s in sorted(rows)]
mean_u = sum(us) / n
mean_m = sum(margins) / n
var_m = sum((m - mean_m) ** 2 for m in margins) / (n - 1)
se_m = math.sqrt(var_m / n)

rng = random.Random(20261114)
B = 20000
boots = []
for _ in range(B):
    boots.append(sum(us[rng.randrange(n)] for _ in range(n)) / n)
boots.sort()
ci_lo = boots[int(0.025 * B)]
ci_hi = boots[int(0.975 * B)]
p_ge05 = sum(1 for b in boots if b >= 0.5) / B

def side_stats(side):
    sub = [r for r in rows.values() if r["side"] == side]
    k = len(sub)
    return {
        "n": k,
        "n1r_wins": sum(1 for r in sub if r["win"]),
        "n1r_mean_margin": round(sum(r["margin"] for r in sub) / k, 2),
        "n1r_mean_u": round(sum(r["u"] for r in sub) / k, 4),
    }

out = {
    "matchup": "n1r (A, ours) vs Echo3 (B)",
    "tag": "n1r_echo3",
    "seeds": "989000-989199",
    "n": n,
    "err": 0,
    "n1r_wins": sum(1 for r in rows.values() if r["win"]),
    "echo3_wins": sum(1 for r in rows.values() if not r["win"]),
    "draws": sum(1 for r in rows.values() if r["margin"] == 0),
    "n1r_mean_margin": round(mean_m, 2),
    "n1r_margin_se": round(se_m, 2),
    "n1r_mean_u": round(mean_u, 4),
    "echo3_mean_margin": round(-mean_m, 2),
    "echo3_mean_u": round(1 - mean_u, 4),
    "mean_u_cluster_ci95_bootstrap": [round(ci_lo, 4), round(ci_hi, 4)],
    "bootstrap_B": B,
    "p_boot_u_ge_0.5": p_ge05,
    "n1r_seat0": side_stats(0),
    "n1r_seat1": side_stats(1),
    "max_u_recompute_dev": max_dev,
    "final_jsonl_md5": hashlib.md5(raw).hexdigest(),
    "packs": {
        "n1r": "ba7f84c3652ea2cd14a2243f2b1eace2",
        "echo3": "7b7620499411a088cf4dc17c0e929bd7",
        "harness": "354bcd1bc2dc8bc188c813f53d28c893",
    },
}
print(json.dumps(out, indent=2))
with open(os.path.join(SNAP, "n1r_echo3-analysis.json"), "w") as f:
    json.dump(out, f, indent=2)
