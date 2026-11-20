"""NumPy one-turn freight interception forecast from the candidate seat's public rosters."""

import numpy as np

ACTIONS = ("NONE", "UP", "RIGHT", "DOWN", "LEFT")
DELTAS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))
DEST = np.asarray(
    [
        [((i // 15 + dy) % 15) * 15 + (i % 15 + dx) % 15 for dy, dx in DELTAS]
        for i in range(225)
    ]
)
ix = np.arange(225)
rr = ix // 15
cc = ix % 15
dr = np.abs(rr[:, None] - rr[None])
dc = np.abs(cc[:, None] - cc[None])
DIST = np.minimum(dr, 15 - dr) + np.minimum(dc, 15 - dc)
FEATURES = (
    "stay",
    "continue_last_move",
    "focal_distance_gain",
    "onto_focal",
    "within_focal_one",
    "own_home_gain",
    "our_home_gain",
    "onto_enemy_home",
    "onto_our_home",
    "enemy_home_proximity",
    "our_home_proximity",
    "relative_cargo_onto_focal",
    "relative_cargo_within_one",
    "enemy_load_home_gain",
    "enemy_load_stay",
    "phase_home_gain",
    "own_units_at_cell",
    "other_DS_units_at_cell",
    "other_DS_lighter_can_enter",
    "enemy_own_units_can_enter",
    "focal_distance_gain_if_focal_loaded",
    "home_gain_x_endgame",
)


def home_distance(cells, bases):
    return np.min(
        DIST[
            np.asarray(cells)[:, None],
            np.array(list(bases.values()), dtype=int)[None, :],
        ],
        axis=1,
    )


def prepare(players, seat, step, previous_players=None):
    side, enemy = players[seat], players[1 - seat]
    our = side[2]
    their = enemy[2]
    cur = np.arange(225)
    own_home = home_distance(cur, side[1])
    enemy_home = home_distance(cur, enemy[1])
    # All counts here are pre-decision; agents may choose another move.
    occ_us = np.bincount([w[0] for w in our.values()], minlength=225)
    occ_them = np.bincount([w[0] for w in their.values()], minlength=225)
    reach_us = sum((DIST[w[0]] <= 1 for w in our.values()), start=np.zeros(225, int))
    reach_them = sum(
        (DIST[w[0]] <= 1 for w in their.values()), start=np.zeros(225, int)
    )
    return dict(
        our=our,
        their=their,
        enemy_bases=enemy[1],
        candidate_bases=side[1],
        own_home=own_home,
        enemy_home=enemy_home,
        occ_us=occ_us,
        occ_them=occ_them,
        reach_us=reach_us,
        reach_them=reach_them,
        step=step,
        can_enemy_recruit=enemy[0] >= 30 and len(their) < 10,
        can_our_recruit=side[0] >= 30 and len(our) < 10,
        previous={} if previous_players is None else previous_players[1 - seat][2],
    )


def pairs(state, uid):
    pos, load, job = state["our"][uid]
    if job != "COLLECTOR":
        raise ValueError("Expected candidate collector")
    for eid, (ep, ec, role) in state["their"].items():
        if role == "COLLECTOR" and ec <= load and DIST[pos, ep] <= 2:
            yield eid, enemy_action_features(state, uid, eid)


def enemy_action_features(s, uid, eid):
    pos, load, _ = s["our"][uid]
    ep, ec, _ = s["their"][eid]
    dest = DEST[ep]
    distance = DIST[pos, dest]
    gain = DIST[pos, ep] - distance
    own_home = s["enemy_home"][ep] - s["enemy_home"][dest]
    our_home = s["own_home"][ep] - s["own_home"][dest]
    stay = np.array([1, 0, 0, 0, 0])
    previous = s["previous"].get(eid)
    previous_dir = (
        -1
        if previous is None
        else next((k for k, cell in enumerate(DEST[previous[0]]) if cell == ep), -1)
    )
    rel = min(1.0, float(np.log1p(max(0, load - ec))) / 7.0)
    elight = min(1.0, float(np.log1p(ec)) / 7.0)
    fload = min(1.0, float(np.log1p(load)) / 7.0)
    stage = s["step"] / 299
    other_us_light = sum(
        (
            DIST[p, dest] <= 1
            for other, (p, c, r) in s["our"].items()
            if other != uid and r == "COLLECTOR" and c < ec
        ),
        start=np.zeros(5, int),
    )
    return np.column_stack(
        [
            stay,
            np.arange(5) == previous_dir,
            gain,
            distance == 0,
            distance <= 1,
            own_home,
            our_home,
            s["enemy_home"][dest] == 0,
            s["own_home"][dest] == 0,
            1 / (1 + s["enemy_home"][dest]),
            1 / (1 + s["own_home"][dest]),
            rel * (distance == 0),
            rel * (distance <= 1),
            elight * own_home,
            elight * stay,
            stage * own_home,
            np.minimum(3, s["occ_them"][dest] - (dest == ep)) / 3,
            np.minimum(3, s["occ_us"][dest] - (dest == pos)) / 3,
            np.minimum(2, other_us_light) / 2,
            np.minimum(3, s["reach_them"][dest] - 1) / 3,
            fload * gain,
            max(0.0, (s["step"] - 250) / 49) * own_home,
        ]
    ).astype(np.float32)


def forecast(state, uid, weights, recruit_probability):
    """Five per-order probabilities plus uniform control, friendly reach counts and home distances."""
    pos, load, _ = state["our"][uid]
    target = DEST[pos]
    friendly = (
        state["reach_us"][target]
        - 1
        + (
            np.isin(target, list(state["candidate_bases"].values()))
            & state["can_our_recruit"]
        )
    )
    safe = np.ones(5)
    uniform_safe = np.ones(5)
    sources = {}
    for eid, x in pairs(state, uid):
        scores = x @ weights
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        at_e = DEST[state["their"][eid][0]]
        intersection = target[:, None] == at_e[None, :]
        pm = intersection @ probs
        sources[eid] = dict(
            cell=state["their"][eid][0],
            cargo=state["their"][eid][1],
            conditional_action_probabilities=probs.tolist(),
        )
        safe *= 1 - pm
        uniform_safe *= 1 - intersection.sum(axis=1) / 5
    new_collector = (
        np.isin(target, list(state["enemy_bases"].values()))
        & state["can_enemy_recruit"]
    )
    safe *= 1 - new_collector * recruit_probability
    uniform_safe *= 1 - new_collector * recruit_probability
    return dict(
        risk=np.clip(1 - safe, 0, 1).tolist(),
        baseline=np.clip(1 - uniform_safe, 0, 1).tolist(),
        friend=friendly.astype(int).tolist(),
        home=state["own_home"][target].astype(int).tolist(),
        source_distributions=sources,
    )
