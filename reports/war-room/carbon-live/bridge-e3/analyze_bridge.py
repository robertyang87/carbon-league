#!/usr/bin/env python3
"""Echo3 bridge field analysis: Echo3 deployment pack vs Echo2, seeds 986000-986199, double-seat 400 games.
Readout from Echo3 perspective: mean margin / mean U / W-D-L + seed-cluster bootstrap CI95.
Stored line fields (margin/u/win) are A-package perspective; verified by U recompute."""
import json, math, random, hashlib, sys

SNAP = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"
E3 = "7b7620499411a088cf4dc17c0e929bd7"

def U(margin, win):
    w = 1.0 if win else (0.5 if margin == 0 else 0.0)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-margin / 1000.0))

rows = []  # dicts: seed, e3_seat, e3_margin, e3_u, e3_win(1/0/None draw)
max_dev = 0.0
errs = 0
for tag, e3_is_a in (("echo3bridge_a", True), ("echo3bridge_b", False)):
    path = f"{SNAP}/{tag}.jsonl"
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    n = 0
    for line in open(path):
        r = json.loads(line)
        n += 1
        if r["err"] is not None:
            errs += 1
        assert r["hh"] == "354bcd1bc2dc8bc188c813f53d28c893", r["hh"]
        assert (r["ah"] == E3) == e3_is_a, (tag, r["ah"])
        dev = abs(U(r["margin"], r["win"]) - r["u"])
        max_dev = max(max_dev, dev)
        if e3_is_a:
            m, u, w = r["margin"], r["u"], (1 if r["win"] else (0.5 if r["margin"] == 0 else 0))
            e3_seat = r["side"]
        else:
            m, u, w = -r["margin"], 1.0 - r["u"], (0 if r["win"] else (0.5 if r["margin"] == 0 else 1))
            e3_seat = 1 - r["side"]
        rows.append(dict(seed=r["seed"], seat=e3_seat, margin=m, u=u, w=w, tag=tag))
    print(f"{tag}: lines={n} md5={md5}")

assert len(rows) == 400, len(rows)
assert errs == 0, errs
seeds = sorted(set(r["seed"] for r in rows))
assert seeds == list(range(986000, 986200)), (seeds[0], seeds[-1], len(seeds))
for s in seeds:
    assert sum(1 for r in rows if r["seed"] == s) == 2

wins = sum(1 for r in rows if r["w"] == 1)
draws = sum(1 for r in rows if r["w"] == 0.5)
losses = sum(1 for r in rows if r["w"] == 0)
mean_margin = sum(r["margin"] for r in rows) / 400
mean_u = sum(r["u"] for r in rows) / 400

# seed-cluster bootstrap for mean U and mean margin
by_seed = {s: [r for r in rows if r["seed"] == s] for s in seeds}
rng = random.Random(20261115)
B = 20000
us, ms = [], []
for _ in range(B):
    su = sm = cnt = 0
    for _ in range(200):
        g = by_seed[rng.choice(seeds)]
        for r in g:
            su += r["u"]; sm += r["margin"]; cnt += 1
    us.append(su / cnt); ms.append(sm / cnt)
us.sort(); ms.sort()
ci_u = [us[int(0.025 * B)], us[int(0.975 * B)]]
ci_m = [ms[int(0.025 * B)], ms[int(0.975 * B)]]
p_u = sum(1 for x in us if x <= 0.5) / B

def seat_stats(seat):
    g = [r for r in rows if r["seat"] == seat]
    return dict(n=len(g), wins=sum(1 for r in g if r["w"] == 1),
                draws=sum(1 for r in g if r["w"] == 0.5),
                mean_margin=round(sum(r["margin"] for r in g) / len(g), 2),
                mean_u=round(sum(r["u"] for r in g) / len(g), 4))

out = dict(
    matchup="Echo3 deployment package (m3s2r1 int8, 7b762049) vs Echo2 (examiner); readout from Echo3 perspective",
    tags=["echo3bridge_a", "echo3bridge_b"], seeds="986000-986199", n=400, err=0,
    echo3_wins=wins, draws=draws, echo3_losses=losses,
    echo3_mean_margin=round(mean_margin, 2), echo3_mean_u=round(mean_u, 4),
    mean_u_cluster_ci95_bootstrap=[round(ci_u[0], 4), round(ci_u[1], 4)],
    mean_margin_cluster_ci95_bootstrap=[round(ci_m[0], 2), round(ci_m[1], 2)],
    bootstrap_B=B, p_boot_u_le_0p5=round(p_u, 5),
    echo3_seat0=seat_stats(0), echo3_seat1=seat_stats(1),
    max_u_recompute_dev=max_dev,
    packs=dict(echo3=E3, echo2_wrapper="15b6d754e6e21de33870074d3983a16f",
               echo2_weights="82fde46b2633950d3cbb29e8cacbfdc2",
               harness="354bcd1bc2dc8bc188c813f53d28c893"),
    bridge_baseline_note="M3s3z ladder profile vs Echo2 +2901.0 (n=4) is the other end of the bridge; this field tightens our-side error only.",
)
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open("/home/hatch/workspace/carbon-live/bridge-e3/echo3bridge-analysis.json", "w"), indent=1, ensure_ascii=False)
