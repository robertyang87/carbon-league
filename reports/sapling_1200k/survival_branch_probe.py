"""Collect on-policy immediate extinction labels, conditioned on a scripted opponent's action.

Run with one BLAS thread, e.g. taskset -c 4 python survival_branch_probe.py OUT.
OUT must be new. No candidate weight or gameplay action is modified.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from eval_v53 import CFG, EPISODE_STEPS, _extinct
from fastsim import FastCarbon

PACK = ROOT / "data/weights/sapling_1200k/agent_int8.py"
OPPONENT = ROOT / "code/policy_raid_v0_1.py"
SEEDS = tuple(range(620089740, 620089744))
PUBLIC_FEATURES = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo",
                   "hist_cell", "hist_cargo", "hist_move")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(feat_module, features, state, actions):
    prefix = feat_module.Prefix(state, features.aux)
    y = features.labels(actions)
    uf = np.zeros((feat_module.OWN, feat_module.UF), np.float16)
    mf = np.zeros((feat_module.OWN, 5, feat_module.MF), np.float16)
    legal = np.zeros((feat_module.OWN, 5), bool)
    for slot in range(feat_module.OWN):
        if state["tok_mask"][slot]:
            uf[slot], mf[slot], legal[slot] = prefix.feats(slot)
            if not legal[slot, y[slot]]:
                raise ValueError("Logged action failed the prefix's legality check")
            prefix.advance(slot, int(y[slot]))
    return {**{k: state[k] for k in PUBLIC_FEATURES},
            "uf": uf, "mf": mf, "legal": legal, "logged_prefix_actions": y}


def classify(logged, extinct):
    if extinct[logged]:
        return "all_actions_extinct" if all(extinct) else "logged_extinct_alternative_survives"
    return "logged_survives_alternative_extinct" if any(extinct) else "all_actions_survive"


def play(seed, seat, pack, opponent_module, feat_module, arrays):
    random.seed(seed)
    np.random.seed(seed)
    sim = FastCarbon(CFG).reset(seed=seed)
    branch_sim = FastCarbon(CFG)
    features = feat_module.Feat(seat)
    agents = [None, None]
    agents[seat] = pack.agent
    agents[1-seat] = opponent_module.make_agent()
    recruit = min(CFG["recCollectorCost"], CFG["recPlanterCost"])
    risk = Counter()
    categories = Counter()
    examples = {}
    trace = []
    for step in range(EPISODE_STEPS-1):
        observations = [sim.obs(i) for i in range(2)]
        state = features.observe(observations[seat])
        actions = [agent(observations[i], sim.config) for i, agent in enumerate(agents)]
        cash, _, workers, trees = observations[seat]["players"][seat]
        n_workers, n_trees = len(workers), len(trees)
        trace.append(dict(step=step, workers=n_workers, trees=n_trees, cash=round(cash, 2)))
        trace = trace[-4:]
        expected_post = None
        if step > 10 and not trees and cash < recruit and 1 <= n_workers <= 2:
            risk[f"own_workers_{n_workers}"] += 1
            if n_workers == 1:
                uid = features.own_uid[0]
                assert uid in workers and state["tok_mask"][0]
                fields = encode(feat_module, features, state, actions[seat])
                assert fields["legal"][0].all()
                logged = int(fields["logged_prefix_actions"][0])
                original = sim.get_state()
                extinct = []
                for candidate in range(5):
                    substituted = fields["logged_prefix_actions"].copy()
                    substituted[0] = candidate
                    physical = features.to_actions(substituted)
                    # The entire remaining joint action, including our bases, stays fixed.
                    command = dict(actions[seat])
                    if uid in physical:
                        command[uid] = physical[uid]
                    else:
                        command.pop(uid, None)
                    branch_actions = actions.copy()
                    branch_actions[seat] = command
                    branch_sim.set_state(original)
                    branch_sim.step(*branch_actions)
                    post = branch_sim.get_state()
                    extinct.append(bool(_extinct(post, seat)))
                    if candidate == logged:
                        expected_post = post
                category = classify(logged, extinct)
                categories[category] += 1
                if category not in examples:
                    examples[category] = dict(seed=seed, seat=seat, step=step, worker=uid,
                        worker_public=workers[uid], preceding_states=list(trace),
                        logged_canonical_action=feat_module.MOVES[logged],
                        extinct_by_canonical_action=dict(zip(feat_module.MOVES, extinct)))
                fields.update(seed=seed, seat=seat, step=step, teacher=pack._OPTS["teacher"],
                              target_slot=0, extinct_after_one_joint_step=extinct)
                for key, value in fields.items():
                    arrays[key].append(value)
        sim.step(*actions)
        actual_post = sim.get_state()
        if expected_post is not None and expected_post != actual_post:
            raise ValueError("Logged action failed exact transition replay")
        if any(_extinct(actual_post, i) for i in range(2)):
            break
    return dict(seed=seed, seat=seat, decisions=step+1, own_terminal_extinction=_extinct(actual_post, seat),
                opponent_terminal_extinction=_extinct(actual_post, 1-seat), normal_limit=not any(
                    _extinct(actual_post, i) for i in range(2)) and sim.step_count == EPISODE_STEPS-1,
                last_worker_root_categories=dict(categories), scarce_root_counts=dict(risk), examples=examples)


def idle_base_runs(arrays):
    """Consecutive public states with the last collector empty, staying at an own base."""
    found = []
    for row, y in enumerate(arrays["logged_prefix_actions"]):
        s = arrays["step"][row]
        seed, seat = arrays["seed"][row], arrays["seat"][row]
        bases = arrays["tok_pos"][row][10:12][arrays["tok_mask"][row][10:12]]
        if not (arrays["tok_kind"][row][0] == 0 and arrays["tok_cargo"][row][0] == 0
                and y[0] == 0 and arrays["tok_pos"][row][0] in bases):
            continue
        if found and (found[-1]["seed"], found[-1]["seat"], found[-1]["last_step"] + 1) == (seed, seat, s):
            found[-1]["last_step"] = s
            found[-1]["decisions"] += 1
        else:
            found.append(dict(seed=seed, seat=seat, first_step=s, last_step=s, decisions=1))
    return found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    pack = load(PACK, "survival_sapling_pack")
    opponent = load(OPPONENT, "survival_scripted_raid")
    features = sys.modules["cnet2_feat"]
    arrays = defaultdict(list)
    results = [play(seed, i % 2, pack, opponent, features, arrays) for i, seed in enumerate(SEEDS)]
    data_path = args.out / "survival_rows.npz"
    np.savez_compressed(data_path, **{k:np.asarray(v) for k,v in arrays.items()})
    labels = np.asarray(arrays["extinct_after_one_joint_step"], bool).reshape(-1, 5)
    report = dict(schema="sapling-conditional-extinction/1",
        scope="Training-only probe. Fixed four distinct maps with alternating seats. All roots after decision step 10 with 1-2 workers, no trees, and cash below recruitment price counted. Every one-worker root enumerated; both the opponent and our bases retain their logged commands. The label measures only next-step candidate extinction conditional on those simultaneous commands. It is not the opponent action distribution, a winning move, or a terminal return.",
        sources={str(p.relative_to(ROOT)):digest(p) for p in [PACK, OPPONENT, ROOT/"code/eval_v53.py",
                ROOT/"code/fastsim.py", Path(__file__).resolve()]},
        embedded_feature_sha256=pack._SRC_SHA256["cnet2_feat"], maps=results,
        feature_contract="The paired targets are stored in canonical move order NONE, UP, RIGHT, DOWN, LEFT. target_slot=0 is the only worker. Inputs use Feat.observe(sim.obs(seat)) at every step, including before the training window; uf/mf/legal use the actor's logged prefix. Only slot 0 has a target and its prefix is empty. Do not feed logged_prefix_actions, map identifiers or label arrays into the risk predictor. No opponent commands or full carbon map are actor inputs. All rows are from the training maps; validation requires disjoint new maps. If comparing two actions use only roots with differing outcomes.",
        idle_empty_collector_base_runs=idle_base_runs(arrays),
        labeled_roots=len(labels), evaluated_actions=int(labels.size), extinct_actions=int(labels.sum()),
        roots_with_both_outcomes=int((labels.any(axis=1) & ~labels.all(axis=1)).sum()),
        train_ready_npz=dict(sha256=digest(data_path), bytes=data_path.stat().st_size,
            arrays={k:list(np.asarray(v).shape) for k,v in arrays.items()}))
    (args.out / "survival_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ["labeled_roots", "evaluated_actions", "extinct_actions",
                                             "roots_with_both_outcomes"]}),flush=True)


if __name__ == "__main__":
    main()
