"""Controlled inference, real teacher/FastCarbon/original interpreter.

No weights, training, business-source edits or complete games. The probe
records original S01 counterexamples after repair plus terminal-input boundary defects; exit zero is not
a statement that the current teacher passed every contract.
"""
import copy
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import types


def find_repo():
    if os.environ.get("CARBON_REPO"):
        root = pathlib.Path(os.environ["CARBON_REPO"]).expanduser().resolve()
        if not (root / "code/stage0_teacher.py").is_file():
            raise RuntimeError("CARBON_REPO must contain code/stage0_teacher.py")
        return root
    for start in (pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()):
        for root in (start, *start.parents):
            if (root / "code/stage0_teacher.py").is_file():
                return root
    raise RuntimeError("Set CARBON_REPO")


ROOT = find_repo()
sys.path.insert(0, str(ROOT / "code/env_src/carbon_challenge-master"))
sys.path.insert(0, str(ROOT / "code"))
import numpy as np
import cnet2_feat as F
import cnet2_search as S
import stage0_teacher as T
from fastsim import FastCarbon, DEFAULT_CONFIG
from zerosum_env import utils
from zerosum_env.envs.carbon import carbon as rules, idgen

LEAF_SENTINEL = 0.876


def utility(margin):
    wdl = 1 if margin > 0 else 0 if margin < 0 else .5
    return .5 * wdl + .5 / (1 + math.exp(-margin / 1000))


class FixedPolicy(S.CachePolicy):
    """No net is loaded. Return exact specified actions and a sentinel U.

    Losing side has a zero-cargo collector; other workers carry positive
    cargo. DOWN has the same meaning in either seat's canonical frame.
    """
    def __init__(self, *, isolate=False, recruit=False, allow_alt=False, recruit_at=100):
        self.isolate, self.recruit, self.allow_alt = isolate, recruit, allow_alt
        self.recruit_at = recruit_at
        self.leaf_calls = []

    def act(self, x, aux, *, want_u=False, forced=None, **kwargs):
        step = 299 - int(x["left"])
        labels = np.where(x["tok_mask"][:F.OWN], 0, -1).astype(np.int64)
        for s in range(10):
            if labels[s] >= 0 and int(x["tok_kind"][s]) == 0 and x["tok_cargo"][s] == 0:
                if not self.isolate or step == 100:
                    labels[s] = F.MOVES.index("DOWN")
        if self.recruit and x["cash"] >= 50 and step == self.recruit_at:
            labels[10] = 1
        logits = np.full((F.OWN, 5), -1e9)
        for s, action in enumerate(labels):
            if action >= 0:
                logits[s, action] = 0
                if (self.isolate or self.allow_alt) and step == 100 and action == F.MOVES.index("DOWN"):
                    logits[s, 0] = -1
        if forced is not None:
            labels = np.where(np.asarray(forced) >= 0, forced, labels)
        if want_u:
            self.leaf_calls.append(dict(step=step, alive=int(x["tok_mask"][:10].sum())))
            return labels, logits, 0., LEAF_SENTINEL
        return labels, logits, 0.


class RecordingSim(FastCarbon):
    instances = []
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trace = []
        self.instances.append(self)

    def step(self, a, b):
        pre = self.get_state()
        super().step(a, b)
        self.trace.append(dict(pre=pre, actions=copy.deepcopy([a, b]), post=self.get_state()))
        return self


def scene(me=0, step=100, distance=3, isolate=False, recruit=False):
    sim = FastCarbon().reset(full_carbon=[0] * 225)
    obs = sim.full_observation()
    obs["step"] = step
    for p in (0, 1):
        obs["players"][p][0] = 50 if recruit and p != me else 20
    target = sim.bases[1-me][1][1]
    obs["players"][me][2] = {f"player-{me}-worker-0": [int(F.DEST[target, 1] if distance == 1 else (target - 15*distance) % 225), 0, "COLLECTOR"]}
    if isolate:
        obs["players"][me][2][f"player-{me}-worker-1"] = [10, 1, "COLLECTOR"]
    obs["players"][1-me][2] = {f"player-{1-me}-worker-0": [200, 7, "COLLECTOR"]}
    sim.load_observation(obs)
    fm, fo = F.Feat(me), F.Feat(1-me)
    # Give roots actual multi-row history; continuation code must not mutate it.
    for f, seat in ((fm, me), (fo, 1-me)):
        f.observe(dict(sim.obs(seat), step=step-1))
    xm, xo = fm.observe(sim.obs(me)), fo.observe(sim.obs(1-me))
    return sim, xm, xo, fm, fo


def original_reference(root_obs, trace):
    states = []
    for p in (0, 1):
        o = copy.deepcopy(root_obs)
        o["player"] = p
        o["full_carbon"] = list(o["carbon"])
        states.append(utils.structify(dict(observation=o, action={}, status="ACTIVE",
                                            reward=float(root_obs["players"][p][0]))))
    idgen.reset()
    for p in (0, 1):
        count = max((int(w.rsplit("-", 1)[-1])+1 for w in root_obs["players"][p][2]), default=0)
        for _ in range(count):
            idgen.new_worker_id(p)
    steps = 0
    for entry in trace:
        if states[0].observation.step >= 299 or any(s.status == "DONE" for s in states):
            break
        for p in (0, 1):
            states[p].action = entry["actions"][p]
        states = rules.compute_next_board(states, types.SimpleNamespace(configuration=utils.structify(DEFAULT_CONFIG)))
        steps += 1
    return dict(scores=[float(s.reward) for s in states], step=int(states[0].observation.step),
                transitions=steps, extinct=any(s.status == "DONE" for s in states))


def run_terminal_case(name, *, me=0, step=100, distance=3, H=2, recruit=False, recruit_at=100):
    sim, xm, xo, fm, fo = scene(me, step, distance, recruit=recruit)
    original_state = copy.deepcopy(sim.get_state())
    pol = FixedPolicy(recruit=recruit, recruit_at=recruit_at)
    RecordingSim.instances = []
    old = T.FastCarbon
    T.FastCarbon = RecordingSim
    try:
        result = T.evaluate_root(sim, me, pol, xm, fm.aux, xo, fo.aux, fm, fo,
                                 K=1, M=1, H=H, guard="on")
    finally:
        T.FastCarbon = old
    assert sim.get_state() == original_state
    clone = RecordingSim.instances[0]
    ref = original_reference(sim.full_observation(), clone.trace)
    expected = utility(ref["scores"][me]-ref["scores"][1-me])
    terminal = any(T._extinct_state(clone.get_state(), p) for p in (0, 1)) or clone.step_count >= 299
    direct = T._terminal_u(clone, me, clone.trace[-1]["pre"]) if terminal else None
    q = float(result["Q_mean"][0])
    return dict(name=name, me=me, H=H, guard="on", root_step=step, simulated_transitions=len(clone.trace),
                simulated_end_step=clone.step_count, exact_reference=ref,
                terminal=terminal, expected_terminal_u=expected if terminal else None,
                direct_terminal_helper=direct, got_q=q, terminal_routing_pass=(abs(q-expected)<1e-12) if terminal else None,
                leaf_calls=pol.leaf_calls, root_not_mutated=True)


def equal(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def run_choice_case():
    sim, xm, xo, fm, fo = scene(distance=3)
    RecordingSim.instances = []
    old = T.FastCarbon
    T.FastCarbon = RecordingSim
    try:
        r = T.evaluate_root(sim, 0, FixedPolicy(allow_alt=True), xm, fm.aux, xo, fo.aux,
                            fm, fo, K=2, M=1, H=2, guard="on")
    finally:
        T.FastCarbon = old
    corrected_qvalues = []
    for clone, value in zip(RecordingSim.instances, r["Q_mean"]):
        terminal = any(T._extinct_state(clone.get_state(), p) for p in (0, 1))
        ref = original_reference(sim.full_observation(), clone.trace)
        corrected_qvalues.append(utility(ref["scores"][0]-ref["scores"][1]) if terminal else float(value))
    values = np.array(corrected_qvalues)
    corrected_target = np.exp((values-values.max())/.05) * np.sqrt(r["priors"])
    corrected_target /= corrected_target.sum()
    assert int(np.argmax(r["q"])) == int(np.argmax(corrected_target)) == 1
    assert np.allclose(r["Q_mean"], corrected_qvalues, rtol=0, atol=1e-12)
    return dict(guard="on", H=2, K=2, M=1,
                arms=[fm.to_actions(a) for a in r["arms"]],
                actual_Q=r["Q_mean"].tolist(), actual_q=r["q"].tolist(),
                expected_Q_with_exact_terminal=corrected_qvalues,
                expected_q_with_exact_terminal=corrected_target.tolist(),
                actual_selected_arm=int(np.argmax(r["q"])), expected_selected_arm=1,
                terminal_by_arm=[any(T._extinct_state(s.get_state(), p) for p in (0, 1)) for s in RecordingSim.instances])


def run_isolation(me):
    sim, xm, xo, fm, fo = scene(me=me, distance=1, isolate=True)
    before = copy.deepcopy((fm.__dict__, fo.__dict__, sim.get_state()))
    events, restores = [], []
    real_guard, real_feat = T.BankruptcyGuard, F.Feat
    features = []
    class TracedGuard(real_guard):
        def restore(self, snap):
            super().restore(snap)
            restores.append(dict(instance=id(self), lostw=self.lostw, prevw=sorted(self.prevw),
                                 aliases_snapshot=(self.prevw is snap["prevw"] or self.guard_base is snap["guard_base"])))
        def apply(self, x, aux, y, lg, pol, own_uid, teacher, stats=None):
            prev = sorted(self.prevw)
            out = super().apply(x, aux, y, lg, pol, own_uid, teacher, stats)
            own = [u for u in own_uid[:10] if u is not None]
            seat = int(own[0].split("-")[1]) if own else None
            events.append(dict(instance=id(self), step=299-int(x["left"]), seat=seat,
                               before_prevw=prev, after_prevw=sorted(self.prevw), lostw=self.lostw))
            return out
    class TracedFeat(real_feat):
        def __init__(self, seat):
            super().__init__(seat)
            features.append(self)
    T.BankruptcyGuard, F.Feat = TracedGuard, TracedFeat
    try:
        r = T.evaluate_root(sim, me, FixedPolicy(isolate=True), xm, fm.aux, xo, fo.aux,
                            fm, fo, K=2, M=2, H=2)
    finally:
        T.BankruptcyGuard, F.Feat = real_guard, real_feat
    assert r["n_arms"] == 2
    assert equal(before, (fm.__dict__, fo.__dict__, sim.get_state()))
    assert all(not row["aliases_snapshot"] for row in restores)
    assert all(row["lostw"] == 0 for row in restores)
    for row in events:
        assert all(int(uid.split("-")[1]) == row["seat"] for uid in row["before_prevw"]), row
    assert sorted({row["lostw"] for row in events if row["step"] > 100 and row["seat"] == me}) == [0, 1]
    assert len({id(f) for f in features}) == len(features)
    return dict(me=me, arms=r["n_arms"], scenarios=2, pass_=True,
                root_and_features_not_mutated=True, no_cross_seat_snapshots=True,
                no_snapshot_set_aliases=True, lostw_independent_across_branches=True,
                fresh_feature_instances=len(features), history_lengths=sorted({len(h) for f in features for h in f.hist.values()}),
                events=events, restores=restores)


def run_extinct_root(me):
    sim, xm, xo, fm, fo = scene(me=me)
    obs = sim.full_observation()
    obs["players"][me][2] = {}
    sim.load_observation(obs)
    fm, fo = F.Feat(me), F.Feat(1-me)
    xm, xo = fm.observe(sim.obs(me)), fo.observe(sim.obs(1-me))
    before = copy.deepcopy(sim.get_state())
    pol = FixedPolicy()
    try:
        T.evaluate_root(sim, me, pol, xm, fm.aux, xo, fo.aux, fm, fo,
                        K=1, M=1, H=2, guard="on")
    except ValueError as exc:
        assert sim.get_state() == before
        return {"me": me, "rejected": True, "error": str(exc), "root_not_mutated": True}
    raise AssertionError("already-extinct root was not rejected")


cases = []
for me in (0, 1):
    cases.append(run_terminal_case("root_extinction_survivor_spends", me=me, distance=1, recruit=True))
    cases.append(run_terminal_case("early_continuation_extinction", me=me, distance=2))
    cases.append(run_terminal_case("last_continuation_extinction", me=me, distance=3))
cases += [run_terminal_case("H1_last_continuation_extinction", H=1, distance=2),
          run_terminal_case("H0_root_extinction", H=0, distance=1),
          run_terminal_case("early_continuation_survivor_spends", distance=2, recruit=True, recruit_at=101),
          run_terminal_case("last_continuation_survivor_spends", distance=3, recruit=True, recruit_at=102),
          run_terminal_case("last_legal_root", step=298, distance=10),
          run_terminal_case("already_timed_out_root", step=299, distance=10),
          run_terminal_case("nonterminal_H2_depth", distance=10)]
cases += [run_terminal_case("H1_last_continuation_extinction", me=1, H=1, distance=2),
          run_terminal_case("H0_root_extinction", me=1, H=0, distance=1),
          run_terminal_case("early_continuation_survivor_spends", me=1, distance=2, recruit=True, recruit_at=101),
          run_terminal_case("last_continuation_survivor_spends", me=1, distance=3, recruit=True, recruit_at=102),
          run_terminal_case("last_legal_root", me=1, step=298, distance=10),
          run_terminal_case("already_timed_out_root_survivor_spends", me=0, step=299, distance=10, recruit=True, recruit_at=299),
          run_terminal_case("already_timed_out_root_survivor_spends", me=1, step=299, distance=10, recruit=True, recruit_at=299)]
for row in cases:
    if row["terminal"] and row["root_step"] < 299:
        assert abs(row["direct_terminal_helper"] - row["expected_terminal_u"]) < 1e-12, row
assert all(r["terminal_routing_pass"] for r in cases if r["name"] in ("root_extinction_survivor_spends", "early_continuation_extinction", "early_continuation_survivor_spends", "last_legal_root"))
assert all(r["terminal_routing_pass"] for r in cases if r["name"] in ("last_continuation_extinction", "last_continuation_survivor_spends", "H1_last_continuation_extinction", "H0_root_extinction"))
out = {"commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
       "terminal_cases": cases, "choice_consequence": run_choice_case(),
       "guard_isolation": [run_isolation(0), run_isolation(1)],
       "already_extinct_root": [run_extinct_root(0), run_extinct_root(1)],
       "source_s01_closed": all(r["terminal_routing_pass"] for r in cases if r["terminal"] and r["root_step"] < 299),
       "timed_out_root_advanced": any(r["simulated_transitions"] > 0 for r in cases if r["root_step"] >= 299),
       "scope": "Real teacher/FastCarbon/original interpreter, fixed policy inference only; no training or real-model performance result."}
base = os.environ.get("CARBON_REVIEW_OUT")
if base:
    pathlib.Path(base).expanduser().mkdir(parents=True, exist_ok=True)
    base = str(pathlib.Path(base).expanduser().resolve())
dest = pathlib.Path(tempfile.mkdtemp(prefix="teacher-terminal-", dir=base)) / "results.json"
dest.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"evidence": str(dest), "terminal_cases": cases, "choice_consequence": out["choice_consequence"],
                  "isolation_pass": [r["pass_"] for r in out["guard_isolation"]]}, indent=2))
