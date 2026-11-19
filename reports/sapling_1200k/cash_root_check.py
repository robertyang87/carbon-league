"""Check one Echo4 relabel versus the Sapling V2 action on untouched V2 trajectories.

Run with --out /tmp/cash_root_check.json. These three maps are reserved for this
diagnostic. Each branch resumes the greedy V2 after one joint action override.
"""
import argparse
import base64
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import zlib

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "reports/sapling_bootstrap"))
import cnet2_feat as F
import cnet2_np as NP
from eval_v53 import CFG, _extinct
from fastsim import FastCarbon
import policy_v0_1 as practice
import smoke_ab as shared

STUDENT = ROOT / "data/weights/sapling_1200k/top5_h2h_v2_agent.py"
TEACHER = ROOT / "code/packs/echo4_int8.py"
SEEDS = (620098011, 620098012, 620098013)
WINDOWS = ((20, 41), (42, 63), (64, 85))
HORIZON = 12


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def get_policy(package, key, dims, metadata):
    blob = zlib.decompress(base64.b64decode(getattr(package, key)))
    weights = package._dequantize(blob, getattr(package, dims))
    weights["meta"] = np.array(getattr(package, metadata))
    return NP.Policy(weights)


def advance(sim, seat, ours, theirs, ledger=None):
    acts = (ours, theirs) if seat == 0 else (theirs, ours)
    if ledger is not None:
        ledger.step(sim, acts)
    else:
        sim.step(*acts)
    post = sim.get_state()
    return post, [_extinct(post, p) for p in (0, 1)]


def rollout(source, seat, root_act, opposing_root_act, feat_at_root, opponent_at_root, student):
    sim = copy.deepcopy(source)
    feat = copy.deepcopy(feat_at_root)
    opp = copy.deepcopy(opponent_at_root)
    ledger = shared.Observer(FastCarbon)
    ours, theirs = root_act, opposing_root_act
    for depth in range(HORIZON):
        if depth:
            x = feat.observe(sim.obs(seat))
            ours = feat.to_actions(student.act(x, feat.aux, teacher=2)[0])
            theirs = opp.act(sim.obs(1-seat), sim.config)
        post, extinct = advance(sim, seat, ours, theirs, ledger)
        if any(extinct):
            break
    return dict(end_step=sim.step_count, end_cash=float(sim.cash[seat]),
                opponent_end_cash=float(sim.cash[1-seat]), extinct_by_seat=extinct,
                positive_delivery_events=int(ledger.d[seat]["positive_delivery_events"]),
                own_positive_delivery_carbon=float(ledger.d[seat]["own_positive_delivery_carbon"]),
                end_state_sha256=digest(post))


def game(seed, seat, student, teacher, released):
    np.random.seed(seed)
    random.seed(7919 * seed + 104729 * seat + 31)
    sim = FastCarbon(CFG).reset(seed=seed)
    feat = F.Feat(seat)
    opp = practice.Policy()
    results, used_windows, traces = [], set(), {}
    counts = Counter()
    last_step = WINDOWS[-1][1] + HORIZON
    while sim.step_count < last_step:
        t = sim.step_count
        observation = sim.obs(seat)
        x = feat.observe(observation)
        own = feat.to_actions(student.act(x, feat.aux, teacher=2)[0])
        if own != released.agent(observation, sim.config):
            raise ValueError("greedy V2 decode differs from the released agent")
        enemy = opp.act(sim.obs(1-seat), sim.config)
        window = next((i for i, (a, b) in enumerate(WINDOWS) if a <= t <= b), None)
        if window is not None:
            counts["steps_in_windows"] += 1
            cash, _, units, _ = observation["players"][seat]
            max_cargo = max((float(u[1]) for u in units.values() if u[2] == "COLLECTOR"), default=0.)
            if cash < 120 and max_cargo >= 30:
                counts["eligible_steps"] += 1
                proposed = feat.to_actions(teacher.act(x, feat.aux, teacher=2)[0])
                if proposed == own:
                    counts["no_disagreement"] += 1
                elif window in used_windows:
                    counts["later_disagreements_in_used_window"] += 1
                else:
                    used_windows.add(window)
                    counts["selected_disagreements"] += 1
                    repetition = dict(seed=seed, seat=seat, step=t, initial_cash=float(cash),
                                      max_collector_cargo=max_cargo, root_state_sha256=digest(sim.get_state()),
                                      student_joint_action=own, teacher_joint_action=proposed,
                                      shared_opposing_joint_action=enemy)
                    python_rng = random.getstate()
                    numpy_rng = np.random.get_state()
                    try:
                        for name, action in (("student", own), ("teacher", proposed)):
                            random.setstate(python_rng)
                            np.random.set_state(numpy_rng)
                            repetition[name] = rollout(sim, seat, action, enemy, feat, opp, student)
                    finally:
                        random.setstate(python_rng)
                        np.random.set_state(numpy_rng)
                    results.append(repetition)
        post, died = advance(sim, seat, own, enemy)
        traces[sim.step_count] = digest(post)
        if any(died):
            break
    for item in results:
        branch = item["student"]
        if branch["end_state_sha256"] != traces.get(branch["end_step"]):
            raise ValueError("student branch did not reproduce untouched trajectory")
    return dict(seed=seed, seat=seat, main_end=sim.step_count, counts=counts, roots=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error("choose a fresh output")
    release = shared.module(STUDENT, "cash_root_v2")
    echo = shared.module(TEACHER, "cash_root_echo")
    student = get_policy(release, "_W64", "_QMETA", "_WMETA")
    teacher = get_policy(echo, "_B1", "_Q1", "_M1")
    games = [game(seed, side, student, teacher, release) for seed in SEEDS for side in (0, 1)]
    roots = [r for g in games for r in g["roots"]]
    for r in roots:
        a, b = r["student"], r["teacher"]
        r["teacher_minus_student"] = {
            "end_cash": b["end_cash"] - a["end_cash"],
            "end_margin": (b["end_cash"] - b["opponent_end_cash"]) - (a["end_cash"] - a["opponent_end_cash"]),
            "delivery": b["own_positive_delivery_carbon"] - a["own_positive_delivery_carbon"],
            "same_end_step": a["end_step"] == b["end_step"],
        }
    paths = (Path(__file__), STUDENT, TEACHER, Path(F.__file__), Path(NP.__file__),
             ROOT / "code/fastsim.py", Path(practice.__file__), Path(shared.__file__))
    out = dict(scope="Training-independent diagnostic, original greedy V2 vs policy_v0_1. "
               "First discordance per fixed window with own cash <120 and at least one collector carrying >=30. "
               "One root joint action changed, both branches then use V2 and the same responsive opponent. "
               "Terminal branches may end early. Distinct roots in one game are correlated.",
               seeds=SEEDS, windows=WINDOWS, horizon=HORIZON, source_sha256={str(f.relative_to(ROOT)): shared.sha(f) for f in paths},
               procedure="Full simulator, observed student history and opponent policy state are deep-copied after their root observations. "
                         "Each V2 branch end-state hash matches the untouched main trajectory. The teacher sees public actor observations only.",
               games=games)
    with args.out.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write("\n")
    print("roots", len(roots), "from games", sum(bool(g["roots"]) for g in games), flush=True)


if __name__ == "__main__":
    main()
