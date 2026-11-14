#!/usr/bin/env python3
"""KataGo probe2 three-leg field analysis (PREREG rev2), C1/spot-29.
Seeds 985000-985199, double-seat 200 per direction per leg; readout from vleaf_value perspective.
Legs: attr (vs vleaf_const), duel (vs Echo3 deployment pack), ref (vs Master2 wrapper).
Stored line fields (margin/u/win) are A-package perspective; U is zero-sum (Ua+Ub=1), verified by recompute.
Per leg: W-D-L, mean margin, mean U, deltaU = 2*meanU-1 (subject minus opponent), seed-cluster bootstrap CI95.
Also per-step timing from row fields sa/sb (per-side total step ms) and ms (per-step ms) when present."""
import json, math, random, hashlib

SNAP = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"
V = "e8d0b426972103f502b1bc8074b007ec"
HARNESS = "354bcd1bc2dc8bc188c813f53d28c893"
LEGS = {
    "attr": dict(opp="36bbe1e4b0d3b2e20c5626517fe13ae8", opp_name="vleaf_const (constant leaf, M_eq=0)",
                 tags=("kgattr_a", "kgattr_b")),
    "duel": dict(opp="7b7620499411a088cf4dc17c0e929bd7", opp_name="Echo3 deployment pack (production CONF)",
                 tags=("kgduel_a", "kgduel_b")),
    "ref": dict(opp="2aebfbc1008d3475ece8e178696e292d", opp_name="Master2 wrapper (opp_master2r)",
                tags=("kgref_a", "kgref_b")),
}

def U(margin, win):
    w = 1.0 if win else (0.5 if margin == 0 else 0.0)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-margin / 1000.0))

def analyze(leg, cfg):
    rows = []
    max_dev = 0.0
    errs = 0
    file_md5 = {}
    step_ms = {"value": [], "opp": []}
    for tag, v_is_a in ((cfg["tags"][0], True), (cfg["tags"][1], False)):
        path = f"{SNAP}/{tag}.jsonl"
        file_md5[tag] = hashlib.md5(open(path, "rb").read()).hexdigest()
        n = 0
        for line in open(path):
            r = json.loads(line)
            n += 1
            if r["err"] is not None:
                errs += 1
            assert r["hh"] == HARNESS, (tag, r["hh"])
            assert (r["ah"] == V) == v_is_a, (tag, r["ah"])
            assert (r["bh"] if v_is_a else r["ah"]) == cfg["opp"], (tag, "opp md5")
            dev = abs(U(r["margin"], r["win"]) - r["u"])
            max_dev = max(max_dev, dev)
            if v_is_a:
                m, u, w = r["margin"], r["u"], (1 if r["win"] else (0.5 if r["margin"] == 0 else 0))
                v_seat = r["side"]
                if "sa" in r: step_ms["value"].append(r["sa"])
                if "sb" in r: step_ms["opp"].append(r["sb"])
            else:
                m, u, w = -r["margin"], 1.0 - r["u"], (0 if r["win"] else (0.5 if r["margin"] == 0 else 1))
                v_seat = 1 - r["side"]
                if "sb" in r: step_ms["value"].append(r["sb"])
                if "sa" in r: step_ms["opp"].append(r["sa"])
            rows.append(dict(seed=r["seed"], seat=v_seat, margin=m, u=u, w=w, tag=tag))
        assert n == 200, (tag, n)
    assert len(rows) == 400 and errs == 0, (len(rows), errs)
    seeds = sorted(set(r["seed"] for r in rows))
    assert seeds == list(range(985000, 985200)), (seeds[0], seeds[-1], len(seeds))
    for s in seeds:
        assert sum(1 for r in rows if r["seed"] == s) == 2

    wins = sum(1 for r in rows if r["w"] == 1)
    draws = sum(1 for r in rows if r["w"] == 0.5)
    losses = sum(1 for r in rows if r["w"] == 0)
    mean_margin = sum(r["margin"] for r in rows) / 400
    mean_u = sum(r["u"] for r in rows) / 400

    by_seed = {s: [r for r in rows if r["seed"] == s] for s in seeds}
    rng = random.Random(20261115)
    B = 20000
    us, ms_ = [], []
    for _ in range(B):
        su = sm = cnt = 0
        for _ in range(200):
            g = by_seed[rng.choice(seeds)]
            for r in g:
                su += r["u"]; sm += r["margin"]; cnt += 1
        us.append(su / cnt); ms_.append(sm / cnt)
    us.sort(); ms_.sort()
    ci_u = [us[int(0.025 * B)], us[int(0.975 * B)]]
    ci_m = [ms_[int(0.025 * B)], ms_[int(0.975 * B)]]
    p_u = sum(1 for x in us if x <= 0.5) / B

    def seat_stats(seat):
        g = [r for r in rows if r["seat"] == seat]
        return dict(n=len(g), wins=sum(1 for r in g if r["w"] == 1),
                    draws=sum(1 for r in g if r["w"] == 0.5),
                    mean_margin=round(sum(r["margin"] for r in g) / len(g), 2),
                    mean_u=round(sum(r["u"] for r in g) / len(g), 4))

    def step_stats(vals):
        if not vals:
            return None
        s = sorted(vals)
        return dict(games=len(s), mean_total=round(sum(s) / len(s), 1),
                    p95_total=round(s[int(0.95 * len(s))], 1), max_total=round(s[-1], 1))

    return dict(
        leg=leg, opponent=cfg["opp_name"], opponent_md5=cfg["opp"],
        tags=list(cfg["tags"]), file_md5=file_md5, seeds="985000-985199", n=400, err=0,
        value_wins=wins, draws=draws, value_losses=losses,
        value_mean_margin=round(mean_margin, 2), value_mean_u=round(mean_u, 4),
        delta_u=round(2 * mean_u - 1, 4),
        delta_u_cluster_ci95=[round(2 * ci_u[0] - 1, 4), round(2 * ci_u[1] - 1, 4)],
        mean_u_cluster_ci95_bootstrap=[round(ci_u[0], 4), round(ci_u[1], 4)],
        mean_margin_cluster_ci95_bootstrap=[round(ci_m[0], 2), round(ci_m[1], 2)],
        bootstrap_B=B, p_boot_u_le_0p5=round(p_u, 5),
        value_seat0=seat_stats(0), value_seat1=seat_stats(1),
        step_time_totals_sa_sb=dict(value=step_stats(step_ms["value"]), opp=step_stats(step_ms["opp"])),
        max_u_recompute_dev=max_dev,
    )

out = dict(
    matchup="KataGo probe2 vleaf_value (frozen subject 4a3b5f1b lineage; pack e8d0b426) three-leg field, PREREG rev2",
    subject_pack=V, harness=HARNESS, eval_v53="45792dfbea3101812836fe77e6752706",
    legs={leg: analyze(leg, cfg) for leg, cfg in LEGS.items()},
)
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open("/home/hatch/workspace/carbon-live/katago3/katago3-analysis.json", "w"), indent=1, ensure_ascii=False)
