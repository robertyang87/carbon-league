"""Pool-discrimination probe paired analysis (C1/spot-29).

Candidate: it95 frozen net vs m3s2g1 (tag probe_it95_g1).
Baseline: m3s2r1 pre-U frozen vs m3s2g1 (tag probe_base_g1).
Seeds 990000-990199, paired per seed. Cluster bootstrap CI B=20000.
Reads the local snapshot files (pulled off-machine by the snaploop).
"""
import hashlib
import json
import math
import os
import random

SNAP = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"


def u_of(margin):
    w = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-margin / 1000.0))


def load(tag):
    raw = open(os.path.join(SNAP, f"{tag}.jsonl"), "rb").read()
    rows = {}
    for ln in raw.decode().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[r["seed"]] = r
    return raw, rows


cand_raw, cand = load("probe_it95_g1")
base_raw, base = load("probe_base_g1")
assert set(cand) == set(range(990000, 990200)), "candidate seed set"
assert set(base) == set(range(990000, 990200)), "baseline seed set"
assert all(r["err"] is None for r in cand.values())
assert all(r["err"] is None for r in base.values())
max_dev = 0.0
for rows in (cand, base):
    for r in rows.values():
        max_dev = max(max_dev, abs(r["u"] - u_of(r["margin"])))
for name, rows, want_ah in (
    ("candidate", cand, "9e16085acd3549df210c6209ab3fd3cb"),
    ("baseline", base, "11d8005930b298290c2e150949349647"),
):
    for k in ("ah", "bh", "hh"):
        vals = {r[k] for r in rows.values()}
        assert len(vals) == 1, f"{name} {k} not single: {vals}"
    assert {r["ah"] for r in rows.values()} == {want_ah}, f"{name} ah"
    assert {r["bh"] for r in rows.values()} == {"bb997d7137395e6e66f7b03134d96897"}, f"{name} bh"
    assert {r["hh"] for r in rows.values()} == {"354bcd1bc2dc8bc188c813f53d28c893"}, f"{name} hh"
    assert all(r["win"] == (r["margin"] > 0) for r in rows.values()), f"{name} win/margin mismatch"

diffs = []
pos = neg = zero = 0
for s in range(990000, 990200):
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
    "probe": "pool_discrimination_it95_vs_m3s2g1",
    "candidate_tag": "probe_it95_g1",
    "baseline_tag": "probe_base_g1",
    "candidate": "it95 frozen (exp/it95.py md5 9e16085acd3549df210c6209ab3fd3cb)",
    "baseline": "m3s2r1 pre-U frozen (exp/m3s2r1_base.py md5 11d8005930b298290c2e150949349647)",
    "opponent": "m3s2g1 (exp/m3s2g1_base.py blob 79514aeb0c835f3038520ce52baae262f3f3935c, 3728418 B, md5 bb997d7137395e6e66f7b03134d96897)",
    "seeds": "990000-990199",
    "n_pairs": n,
    "candidate_wins": sum(1 for r in cand.values() if r["win"]),
    "baseline_wins": sum(1 for r in base.values() if r["win"]),
    "candidate_mean_margin": round(sum(r["margin"] for r in cand.values()) / n, 2),
    "baseline_mean_margin": round(sum(r["margin"] for r in base.values()) / n, 2),
    "candidate_mean_u": round(sum(r["u"] for r in cand.values()) / n, 4),
    "baseline_mean_u": round(sum(r["u"] for r in base.values()) / n, 4),
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
    "harness_md5": "354bcd1bc2dc8bc188c813f53d28c893",
}
print(json.dumps(out, indent=2))
with open(os.path.join(SNAP, "probe-du.json"), "w") as f:
    json.dump(out, f, indent=2)
