import hashlib
import json
import math
import os
import random
import sys

SNAP = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"


def u_of(margin):
    return 0.5 * (1.0 if margin > 0 else 0.0) + 0.5 / (1.0 + math.exp(-margin / 1000.0))


def load(tag):
    raw = open(os.path.join(SNAP, f"{tag}.jsonl"), "rb").read()
    rows = {}
    for ln in raw.decode().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[r["seed"]] = r
    return raw, rows


out = {}
cand_raw, cand = load("it95_m2")
base_raw, base = load("it95base_m2")
assert set(cand) == set(range(996000, 996200)), "candidate seed set"
assert set(base) == set(range(996000, 996200)), "baseline seed set"
assert all(r["err"] is None for r in cand.values())
assert all(r["err"] is None for r in base.values())
max_dev = 0.0
for rows in (cand, base):
    for r in rows.values():
        max_dev = max(max_dev, abs(r["u"] - u_of(r["margin"])))

diffs = []
pos = neg = zero = 0
for s in range(996000, 996200):
    d = cand[s]["u"] - base[s]["u"]
    diffs.append(d)
    if d > 0:
        pos += 1
    elif d < 0:
        neg += 1
    else:
        zero += 1
n = len(diffs)
mean = sum(diffs) / n
var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
se = math.sqrt(var / n)

rng = random.Random(20261114)
B = 20000
boots = []
for _ in range(B):
    ssum = 0.0
    for _ in range(n):
        ssum += diffs[rng.randrange(n)]
    boots.append(ssum / n)
boots.sort()
ci_lo = boots[int(0.025 * B)]
ci_hi = boots[int(0.975 * B)]
p_le0 = sum(1 for b in boots if b <= 0) / B

out = {
    "candidate_tag": "it95_m2",
    "baseline_tag": "it95base_m2",
    "n_pairs": n,
    "candidate_wins": sum(1 for r in cand.values() if r["win"]),
    "baseline_wins": sum(1 for r in base.values() if r["win"]),
    "candidate_mean_margin": round(sum(r["margin"] for r in cand.values()) / n, 2),
    "baseline_mean_margin": round(sum(r["margin"] for r in base.values()) / n, 2),
    "dU": round(mean, 4),
    "dU_se_paired": round(se, 4),
    "ci95_normal": [round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)],
    "ci95_cluster_bootstrap": [round(ci_lo, 4), round(ci_hi, 4)],
    "bootstrap_B": B,
    "p_boot_le_0": p_le0,
    "pairs_pos_neg_zero": [pos, neg, zero],
    "max_u_recompute_dev": max_dev,
    "candidate_md5": hashlib.md5(cand_raw).hexdigest(),
    "baseline_md5": hashlib.md5(base_raw).hexdigest(),
    "gate_pass": bool(mean >= 0.02 and ci_lo > 0),
}
print(json.dumps(out, indent=2))
with open(os.path.join(SNAP, "it95-du.json"), "w") as f:
    json.dump(out, f, indent=2)
