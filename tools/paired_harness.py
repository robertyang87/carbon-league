"""Paired-game harness for the HOMEFIX2 candidate-internal protection
experiment (rule point homefix2-candidate-internal-v1).

STATUS: PREP ONLY. The experiment does NOT launch from this task.
--run is hard-gated: it refuses unless the environment variable
CARBON_HOMEFIX2_LAUNCH=1 is set AND a frozen registration file is
passed whose contents pin the variant md5, the opponent pack md5 and
the seed block. War-room scheduling + strategy-group registration come
first (RULING_REPLICA3_20261118.md section 1).

Design (per the ruling, pinned verbatim):
  pairs: seeds 964000-964599 (600) x 2 fields = 1,200 pairs;
         per (seed, field): subject seat = seed % 2 on field 0,
         the opposite seat on field 1 (dual-field dual-seat);
         arms = {control: variant with homefix2=False,
                 protection: variant with homefix2=True},
         same (seed, field, seat) for both arms -> paired Delta U.
  opponent: Echo6 composite pack (md5 25f144f3..., path pinned by the
         registration file; probed at --check time).
  metric: (c)-class real-continuation paired Delta U -- U from the
         variant's own _terminal_u (F01/F04 terminal scoring verbatim),
         subject perspective; margin recorded alongside.
  behaviour axes (per game, subject side): unsold = sum of subject
         collectors' cargo in the terminal state; loaded deaths =
         subject collectors that died (sim.last_collisions) while
         carrying cargo > 0 (pre-step cargo), all-game primary,
         left<=40 subset secondary.
  statistics: per-pair diffs; t 95% CI primary (scipy when available,
         normal approx flagged otherwise); bootstrap 10,000 resamples
         (seeded) alongside; disagreement between the two on the
         lower-bound test is NAMED in the summary, never hidden.
  verdict precedence (ruling): CI half-width > 0.02 -> "evidence
         insufficient" FIRST; else point >= 0 and t lower > -0.02 and
         both behaviour axes non-degraded -> PASS (non-inferiority);
         Delta U line met but a behaviour axis degraded -> "mixed";
         else FAIL. Passing is not adoption (separate ruling).
  instrumentation: the protection arm runs with a Homefix2Recorder;
         its summary (trigger rate / adoption rate / delta_q
         distribution) is part of the result file.

Subject form: the annotator's calling form -- in-process with the
true sim, per-seat legal Feats tracked by the referee, belief="sample",
K=16 / M=8 / H=2, teacher=1, weights cnet2_m3s2r1_heads_v2.npz
(md5 06a03d9c...). NOTE for the war room: this differs from the packs'
obs-only actor interface; confirm the exam-seat harness form at
registration.

Usage:
  python3 paired_harness.py --plan
  python3 paired_harness.py --check --registration REG.json
  python3 paired_harness.py --run --registration REG.json   (gated)
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)

SEEDS = list(range(964000, 964600))
CONSUMED_SEED_BLOCKS = [
    (963000, 963599, "rule-layer independent screen (VERDICT_RULE_LAYER_SCREEN_20261118)"),
    (997800, 998399, "F3 composite screen"),
    (998600, 998999, "calibration anchor field"),
    (999000, 999399, "calibration student field"),
    (999400, 999599, "calibration backup field"),
]
T1_SECONDS_PER_ROOT = 16.0  # bfc87903 section 4 registered price, spot-64
DECISIONS_PER_GAME = 299

JUDGMENT_LINE = {
    "pairs": 1200,
    "opponent": "Echo6",
    "delta_u_point_min": 0.0,
    "t_ci_lower_gt": -0.02,
    "half_width_max": 0.02,
    "behaviour_axes_non_degraded": ["unsold_per_game", "loaded_deaths_per_game"],
}


def md5_of(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def load_variant():
    spec = importlib.util.spec_from_file_location(
        "stage0_teacher_homefix2", os.path.join(HERE, "stage0_teacher_homefix2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan_text():
    games = 2 * len(SEEDS) * 2  # pairs(1200) x arms(2)
    roots = games * DECISIONS_PER_GAME
    cpu_h = roots * T1_SECONDS_PER_ROOT / 3600.0
    return {
        "seeds": [SEEDS[0], SEEDS[-1]], "pairs": 2 * len(SEEDS),
        "games_total": games, "subject_decisions_per_game": DECISIONS_PER_GAME,
        "unit_price_s_per_root": T1_SECONDS_PER_ROOT,
        "unit_price_source": "VERDICT_PROBE_B_20261118.md (bfc87903) section 4, T1 = K16/M8/H2, spot-64",
        "cpu_hours_estimate": round(cpu_h, 1),
        "seat_hours_on_64vcpu_estimate": round(cpu_h / 64.0, 1),
        "judgment_line": JUDGMENT_LINE,
    }


def check_registration(reg):
    """Mechanical preflight over the prep artifacts (no games run)."""
    problems = []
    notes = []
    variant = os.path.join(HERE, "stage0_teacher_homefix2.py")
    v_md5 = md5_of(variant)
    if reg.get("variant_md5") != v_md5:
        problems.append(f"variant md5 mismatch: reg {reg.get('variant_md5')} != file {v_md5}")
    if reg.get("base_md5", "").find("c6ade5ce") != 0:
        problems.append("base teacher md5 not pinned to c6ade5ce...")
    if reg.get("rule_point") != "homefix2-candidate-internal-v1":
        problems.append("rule point id mismatch")
    if reg.get("judgment_line") != JUDGMENT_LINE:
        problems.append("judgment line differs from the pinned ruling line")
    seeds = reg.get("seeds")
    if seeds != [SEEDS[0], SEEDS[-1]]:
        problems.append(f"seed block not the registered {SEEDS[0]}-{SEEDS[-1]}")
    else:
        for lo, hi, why in CONSUMED_SEED_BLOCKS:
            if not (SEEDS[-1] < lo or SEEDS[0] > hi):
                problems.append(f"seed block overlaps consumed block {lo}-{hi} ({why})")
    opp = reg.get("opponent_pack") or {}
    path = opp.get("path")
    if not path or not os.path.exists(path):
        problems.append("opponent pack path missing on this seat")
    else:
        if not md5_of(path).startswith(opp.get("md5_prefix", "25f144f3")):
            problems.append("opponent pack md5 does not match the Echo6 registration")
        spec = importlib.util.spec_from_file_location("opponent_pack", path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            entry = next((n for n in ("create_actor", "make_actor", "Actor", "act")
                          if hasattr(mod, n)), None)
            if entry is None:
                problems.append("opponent pack exposes no known actor entry point")
            else:
                notes.append(f"opponent actor entry: {entry}")
        except Exception as e:  # pack import failure is a preflight fact
            problems.append(f"opponent pack import failed: {e}")
    for item in ("five_conditions_registered", "difference_table_filed",
                 "instrumentation_three_readings", "search_pack_verify_gate"):
        if not reg.get(item):
            problems.append(f"precondition not registered: {item}")
    try:
        import scipy  # noqa: F401
        notes.append("scipy available: t CI exact")
    except ImportError:
        notes.append("scipy missing: t CI falls back to normal approx (flagged)")
    return problems, notes


class TeacherSearchActor:
    """The subject: variant teacher search, annotator calling form."""

    def __init__(self, V, pol, homefix2, recorder=None):
        self.V = V
        self.pol = pol
        self.homefix2 = homefix2
        self.recorder = recorder
        self.feat_me = None
        self.feat_opp = None

    def reset(self, seat):
        F = self.V.F
        self.me = seat
        self.feat_me = F.Feat(seat)
        self.feat_opp = F.Feat(1 - seat)

    def act(self, sim):
        V = self.V
        F = V.F
        obs = sim.obs(self.me)
        x_me = self.feat_me.observe(obs)
        x_opp = self.feat_opp.observe(sim.obs(1 - self.me))
        res = V.evaluate_root_homefix2(
            sim, self.me, self.pol, x_me, self.feat_me.aux,
            x_opp, self.feat_opp.aux, self.feat_me, self.feat_opp,
            teacher=1, K=16, M=8, H=2, seed_base=7000, guard="on",
            belief="sample", homefix2=self.homefix2,
            hf2_recorder=self.recorder)
        best = int(max(range(len(res["arms"])), key=lambda i: res["q"][i]))
        return F.Feat.to_actions(self.feat_me, res["arms"][best])


def play_game(V, pol, opponent_act, seed, subject_seat, homefix2, recorder=None):
    """One game. opponent_act(obs_dict_for_opponent_seat) -> action dict.
    Returns the per-game record (subject perspective)."""
    from fastsim import FastCarbon
    sim = FastCarbon()
    sim.reset(seed=seed)
    actor = TeacherSearchActor(V, pol, homefix2, recorder)
    actor.reset(subject_seat)
    loaded_deaths = 0
    loaded_deaths_late = 0
    pre = sim.get_state()
    while not sim.done:
        pre = sim.get_state()
        a_subj = actor.act(sim)
        a_opp = opponent_act(sim.obs(1 - subject_seat))
        if subject_seat == 0:
            sim.step(a_subj, a_opp)
        else:
            sim.step(a_opp, a_subj)
        # last_collisions entries: (wid, player, carbon_at_death, ...)
        for e in getattr(sim, "last_collisions", []):
            if e[1] == subject_seat and float(e[2]) > 0:
                loaded_deaths += 1
                if 299 - sim.step_count <= 40:
                    loaded_deaths_late += 1
    u = V._terminal_u(sim, subject_seat, pre)
    st = sim.get_state()
    cash = [float(st["cash"][0]), float(st["cash"][1])]
    unsold = sum(float(w["carbon"]) for w in st["workers"].values()
                 if w["player"] == subject_seat and w["type"] == "COLLECTOR")
    return {"seed": seed, "subject_seat": subject_seat,
            "homefix2": bool(homefix2), "u": u,
            "margin": cash[subject_seat] - cash[1 - subject_seat],
            "unsold": unsold, "loaded_deaths": loaded_deaths,
            "loaded_deaths_late": loaded_deaths_late}


def summarize(pairs):
    """pairs: list of dicts with du, d_unsold, d_deaths (protection-control).
    Implements the ruling's verdict precedence."""
    import numpy as np
    du = np.array([p["du"] for p in pairs], dtype=np.float64)
    n = len(du)
    mean = float(du.mean())
    sd = float(du.std(ddof=1)) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n else 0.0
    t_crit = 1.96
    t_method = "normal approx (scipy unavailable)"
    try:
        from scipy import stats
        t_crit = float(stats.t.ppf(0.975, n - 1))
        t_method = "scipy t"
    except ImportError:
        pass
    lo, hi = mean - t_crit * se, mean + t_crit * se
    rng = np.random.default_rng(20261119)
    boots = []
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        boots.append(float(du[idx].mean()))
    blo, bhi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    d_unsold = float(np.mean([p["d_unsold"] for p in pairs]))
    d_deaths = float(np.mean([p["d_deaths"] for p in pairs]))
    half = (hi - lo) / 2.0
    disagree = (lo > JUDGMENT_LINE["t_ci_lower_gt"]) != (blo > JUDGMENT_LINE["t_ci_lower_gt"])
    if half > JUDGMENT_LINE["half_width_max"]:
        verdict = "证据不足 (evidence insufficient: CI half-width > 0.02)"
    elif mean >= 0.0 and lo > JUDGMENT_LINE["t_ci_lower_gt"]:
        if d_unsold <= 0 and d_deaths <= 0:
            verdict = "过线 (non-inferiority line met; adoption is a separate ruling)"
        else:
            verdict = "混合 (Delta U line met, a behaviour axis degraded)"
    else:
        verdict = "未过线 (non-inferiority line not met)"
    return {"n_pairs": n, "delta_u_point": mean, "t_ci": [lo, hi],
            "t_method": t_method, "bootstrap_ci": [blo, bhi],
            "ci_disagreement_on_lower_bound": bool(disagree),
            "half_width": half, "d_unsold_point": d_unsold,
            "d_loaded_deaths_point": d_deaths, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--registration")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.plan:
        print(json.dumps(plan_text(), indent=2, ensure_ascii=False))
        return 0
    if args.check or args.run:
        if not args.registration:
            print("registration file required", file=sys.stderr)
            return 2
        reg = json.load(open(args.registration))
        problems, notes = check_registration(reg)
        for nte in notes:
            print(f"note: {nte}")
        if problems:
            for p in problems:
                print(f"PREFLIGHT FAIL: {p}")
            return 3
        print("PREFLIGHT PASS (all mechanical preconditions hold)")
        if args.check:
            return 0
        if os.environ.get("CARBON_HOMEFIX2_LAUNCH") != "1":
            print("LAUNCH GATE: CARBON_HOMEFIX2_LAUNCH != 1 -- refusing to run. "
                  "The experiment launches only on war-room scheduling after "
                  "registration is frozen.", file=sys.stderr)
            return 4
        raise SystemExit("launch path intentionally not exercised in prep; "
                         "wire the opponent adapter at registration time")
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
