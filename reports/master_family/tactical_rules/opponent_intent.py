"""Learn the next move of each visible enemy worker from our pre-action observation.

Run with Python and NumPy from a repo checkout. The model is a role-specific
linear layer over five candidate actions, followed by softmax. Inference never
receives our proposed move. No top-ladder replay is used to fit or select it.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
from cnet2_feat import DEST, DIST, MOVES, move_between

MOVE_ID = {name: index for index, name in enumerate(MOVES)}
FEATURES = [
    "stay", "repeat_previous_move", "reverse_previous_move", "closer_to_home",
    "loaded_closer_to_home", "destination_home", "loaded_destination_home",
    "friend_at_destination", "opposing_worker_at_destination",
    "light_opposing_collector_can_reach", "opposing_tree_at_destination",
    "friendly_tree_at_destination", "visible_ground_carbon", "ground_visible",
    "loaded_stay", "affordable_tree_capture_here",
]
TRAIN = tuple(range(620093000, 620093008))
TEST = tuple(range(620093008, 620093012))
TRANSFER = tuple(range(620093012, 620093016))


def features(obs, previous=None):
    """Return enemy IDs, role indices and [worker,5,F] action features in world frame."""
    seat = obs["player"]
    mine, enemy = obs["players"][seat], obs["players"][1 - seat]
    if previous is not None and (previous["step"] + 1 != obs["step"] or previous["player"] != seat):
        raise ValueError("previous observation must be the same seat's preceding step")
    old = {} if previous is None else previous["players"][1 - seat][2]
    friendly = [unit[0] for unit in enemy[2].values()]
    opposing = [unit[0] for unit in mine[2].values()]
    home = list(enemy[1].values())
    ground = np.asarray(obs["carbon"])
    visible = np.zeros(len(ground), dtype=bool)
    for position, _, kind in mine[2].values():
        visible[DEST[position, :1] if kind == "COLLECTOR" else DEST[position, 1:]] = True
    lightest = np.full(len(ground), np.inf)
    for position, cargo, kind in mine[2].values():
        if kind == "COLLECTOR":
            lightest[DEST[position]] = np.minimum(lightest[DEST[position]], cargo)
    own_tree = [tree[0] for tree in enemy[3].values()]
    opp_tree = [tree[0] for tree in mine[3].values()]
    ids = sorted(enemy[2])
    x = np.zeros((len(ids), 5, len(FEATURES)), dtype=np.float64)
    roles = np.empty(len(ids), dtype=int)
    for row, uid in enumerate(ids):
        position, cargo, kind = enemy[2][uid]
        roles[row] = int(kind == "PLANTER")
        destinations = DEST[position]
        home_distance = DIST[np.ix_(destinations, home)].min(axis=1)
        closer = home_distance[0] - home_distance
        loaded = min(cargo / 250, 2)
        stay = np.arange(5) == 0
        last = move_between(old[uid][0], position) if uid in old else 5
        reverse = (0, 3, 4, 1, 2, 5)[last]
        is_home = home_distance == 0
        occupied_by_friend = np.isin(destinations, friendly) & ~stay
        on_opp_tree = np.isin(destinations, opp_tree)
        x[row] = np.column_stack((
            stay, np.arange(5) == last, (np.arange(5) == reverse) & ~stay,
            closer, loaded * closer, is_home, loaded * is_home,
            occupied_by_friend, np.isin(destinations, opposing),
            lightest[destinations] <= cargo, on_opp_tree,
            np.isin(destinations, own_tree), np.where(visible[destinations], ground[destinations] / 100, 0),
            visible[destinations], loaded * stay, on_opp_tree & stay & (enemy[0] >= 20),
        ))
    return ids, roles, x


def probabilities(x, roles, weights):
    scores = np.einsum("naf,nf->na", x, weights[roles])
    exps = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


def predict(obs, weights, previous=None):
    """Mapping of worker ID to five probabilities in NONE, UP, RIGHT, DOWN, LEFT order."""
    ids, roles, x = features(obs, previous)
    return dict(zip(ids, probabilities(x, roles, np.asarray(weights))))


def recorded_moves(actions, ids):
    return [MOVE_ID[actions.get(uid) or "NONE"] for uid in ids]


def simulate(seeds, policy, steps):
    from eval_v53 import CFG, _extinct
    from fastsim import FastCarbon
    import policy_v0_5 as reference

    rows = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        sim = FastCarbon(CFG).reset(seed=seed)
        agents = [policy.make_agent(), policy.make_agent()]
        beliefs = [reference.BeliefState(), reference.BeliefState()]
        models = [reference.OpponentModel(b) for b in beliefs]
        prev = [None, None]
        for _ in range(steps):
            obs = [sim.obs(s) for s in (0, 1)]
            inputs = []
            for seat in (0, 1):
                ids, roles, x = features(obs[seat], prev[seat])
                beliefs[seat].update(obs[seat])
                view = reference.BoardView(obs[seat])
                pv = reference.compute_plant_value(beliefs[seat].belief, view)
                heuristic = models[seat].predict(view, pv)
                inputs.append((ids, roles, x, heuristic))
            actions = [agents[s](obs[s], CFG) for s in (0, 1)]
            for seat, (ids, roles, x, heuristic) in enumerate(inputs):
                enemy = obs[seat]["players"][1 - seat][2]
                mine = obs[seat]["players"][seat][2]
                my_positions = [u[0] for u in mine.values()]
                my_moves = recorded_moves(actions[seat], mine)
                own_arrivals = {DEST[pos, a] for pos, a in zip(my_positions, my_moves)}
                labels = recorded_moves(actions[1 - seat], ids)
                for j, uid in enumerate(ids):
                    position = enemy[uid][0]
                    dest = DEST[position]
                    probs = dict(heuristic[uid])
                    danger = bool(my_positions and DIST[position, my_positions].min() <= 2)
                    meetings = np.array([p in own_arrivals for p in dest])
                    rows.append((seed, roles[j], x[j], labels[j], danger, meetings,
                                 [probs.get(int(p), 0.0) for p in dest]))
            sim.step(*actions)
            prev = obs
            state = sim.get_state()
            if sim.step_count >= 299 or any(_extinct(state, s) for s in (0, 1)):
                break
    return tuple(np.asarray(col) for col in zip(*rows))


def fit(data):
    _, roles, x, labels, *_ = data
    w = np.zeros((2, len(FEATURES)))
    first, second = np.zeros_like(w), np.zeros_like(w)
    actual = np.eye(5)[labels]
    for step in range(1, 151):
        residual = probabilities(x, roles, w) - actual
        grad = np.stack([np.einsum("na,naf->f", residual[roles == r], x[roles == r])
                         / np.sum(roles == r) for r in (0, 1)]) + 0.001 * w
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad**2
        w -= 0.06 * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
    counts = np.stack([np.bincount(labels[roles == r], minlength=5) + 1 for r in (0, 1)])
    return w, counts / counts.sum(axis=1, keepdims=True)


def evaluate(data, weights, frequency):
    seeds, roles, x, labels, risk, meetings, rule = data
    learned = probabilities(x, roles, weights)
    actual = meetings[np.arange(len(labels)), labels]
    def metrics(mask):
        subset = {"worker_turns": int(mask.sum()), "games": int(len(np.unique(seeds[mask]))),
                  "actual_meetings": int(actual[mask].sum())}
        for name, proba in (("frequency", frequency[roles]), ("existing_rule", rule), ("learned", learned)):
            if not np.allclose(proba.sum(axis=1), 1):
                raise ValueError("invalid action distribution")
            if not mask.any():
                subset[name] = None
                continue
            p = proba[mask]
            selected = p[np.arange(len(p)), labels[mask]]
            # The own move is revealed ONLY to score a frozen prediction. New recruits excluded.
            meet_p = (p * meetings[mask]).sum(axis=1)
            subset[name] = {"top1": float((p.argmax(axis=1) == labels[mask]).mean()),
                            "action_nll": float(-np.log(np.maximum(selected, 1e-15)).mean()),
                            "meeting_brier": float(((meet_p - actual[mask])**2).mean()),
                            "mean_meeting_probability": float(meet_p.mean())}
        return subset
    return {"all": metrics(np.ones(len(labels), dtype=bool)), "within_two_cells": metrics(risk),
            "per_map_within_two_cells": {str(s): metrics(risk & (seeds == s)) for s in np.unique(seeds)}}


def main():
    import policy_v0_5
    import policy_raid_v0_1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train = simulate(TRAIN, policy_v0_5, 100)
    weights, frequency = fit(train)
    result = {"format": "enemy-intent/1", "status": "diagnostic_only", "action_order": MOVES,
              "role_order": ["COLLECTOR", "PLANTER"], "features": FEATURES, "weights": weights.tolist(),
              "parameters": int(weights.size), "role_frequency": frequency.tolist(),
              "train": {"policy": "policy_v0_5", "seeds": TRAIN, "turns": len(train[0])},
              "steps_per_game_max": 100, "seed_unit": "both seats from the same generated map",
              "metrics_note": "NLL and Brier lower is better. Meetings are attempted arrivals on the same cell as any existing own worker after both submitted actions. Births excluded. Death is not the target. Pooled rows are correlated; per-map results also reported.",
              "inputs": "Own live pre-action masked observation and previous own observation. Submitted actions used as labels even when a worker disappears after transition. No future state, top-player replay or our candidate move enters the layer.",
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), ROOT/'code/fastsim.py', ROOT/'code/cnet2_feat.py',
                                 ROOT/'code/policy_v0_5.py', ROOT/'code/policy_raid_v0_1.py', ROOT/'code/eval_v53.py']}}
    for label, seeds, policy in (("held_out", TEST, policy_v0_5), ("different_policy", TRANSFER, policy_raid_v0_1)):
        result[label] = {"policy": policy.__name__, "seeds": seeds,
                         **evaluate(simulate(seeds, policy, 100), weights, frequency)}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
