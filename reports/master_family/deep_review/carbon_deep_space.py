"""Spatial replay summaries from observed states, without simulating policies.

Public API: analyze_space(steps, seat) -> JSON-serializable dict.
Distances are Manhattan distances on the 15x15 torus. "Enemy zone" means
strictly closer to an enemy base than to an own base (ties are neutral), not
a Cartesian half-board. "Enemy near" means distance <= 2 to an enemy base.
Transitions refer to the destination observation's step; no intent is inferred.
"""
from collections import Counter


def distance(a, b, size=15):
    ar, ac = divmod(int(a), size)
    br, bc = divmod(int(b), size)
    dr, dc = abs(ar - br), abs(ac - bc)
    return min(dr, size - dr) + min(dc, size - dc)


def _nearest(pos, bases):
    return min((distance(pos, b) for b in bases), default=None)


def _point(pos):
    return [int(pos) // 15, int(pos) % 15]


def _state(frame, seat, fallback_step):
    if isinstance(frame, list):
        candidates = [frame[seat], *frame]
    else:
        candidates = [frame]
    for entry in candidates:
        obs = entry.get("observation", entry)
        if "players" in obs and len(obs["players"]) == 2:
            return int(obs.get("step", fallback_step)), obs["players"]
    return None


def analyze_space(steps, seat):
    if seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    first = {name: None for name in (
        "enemy_zone_entry", "enemy_near_entry", "own_tree", "enemy_near_tree",
        "collector_return", "wrap_move")}
    events = []
    recruits = []
    phases = {"opening_0_50": {}, "midgame_51_150": {}}
    raw = {name: {role: Counter() for role in ("COLLECTOR", "PLANTER")}
           for name in phases}
    tree_raw = {name: Counter() for name in phases}
    visits = {name: {role: set() for role in ("COLLECTOR", "PLANTER")}
              for name in phases}
    checkpoints = []
    previous_workers = {}
    previous_trees = {}
    seen_workers = set()
    first_planter = None
    valid_states = 0
    returns = {name: [] for name in phases}

    def record_first(key, event):
        if first[key] is None:
            first[key] = event
            if event["step"] <= 50:
                events.append(dict(event, event=key))

    for index, frame in enumerate(steps):
        state = _state(frame, seat, index)
        if state is None:
            continue
        step, players = state
        valid_states += 1
        own, enemy = players[seat], players[1 - seat]
        own_bases = [int(p) for p in own[1].values()]
        enemy_bases = [int(p) for p in enemy[1].values()]
        workers = own[2]
        phase = "opening_0_50" if step <= 50 else "midgame_51_150" if step <= 150 else None
        for uid, (position, cargo, role) in workers.items():
            pos = int(position)
            do, de = _nearest(pos, own_bases), _nearest(pos, enemy_bases)
            in_enemy = de is not None and do is not None and de < do
            near_enemy = de is not None and de <= 2
            detail = {"step": step, "uid": uid, "role": role, "cell": pos,
                      "rc": _point(pos), "d_own": do, "d_enemy": de,
                      "cargo": round(float(cargo), 2)}
            if uid not in seen_workers:
                seen_workers.add(uid)
                if step <= 50:
                    recruits.append({"step": step, "uid": uid, "role": role,
                                     "cell": pos, "base": pos if pos in own_bases else None})
                if role == "PLANTER" and first_planter is None:
                    first_planter = {"uid": uid, "born_step": step, "born_cell": pos,
                                     "last_seen_by_50": dict(detail) if step <= 50 else None,
                                     "max_own_distance_by_50": do if step <= 50 else None}
            if first_planter and first_planter["uid"] == uid and step <= 50:
                first_planter["last_seen_by_50"] = dict(detail)
                if do is not None:
                    first_planter["max_own_distance_by_50"] = max(
                        first_planter["max_own_distance_by_50"] or 0, do)
            if in_enemy:
                record_first("enemy_zone_entry", dict(detail))
            if near_enemy:
                record_first("enemy_near_entry", dict(detail))
            old = previous_workers.get(uid)
            if old is not None:
                old_pos, old_cargo, _ = old
                if (pos in own_bases and int(old_pos) not in own_bases and role == "COLLECTOR"
                        and float(old_cargo) > 0 and float(cargo) < float(old_cargo)):
                    event = {"step": step, "uid": uid, "base": pos,
                             "from_cell": int(old_pos), "cargo_before": round(float(old_cargo), 2),
                             "cargo_after": round(float(cargo), 2)}
                    record_first("collector_return", event)
                    if phase:
                        returns[phase].append(event)
                if distance(old_pos, pos) == 1 and (abs(int(old_pos) // 15 - pos // 15) == 14
                                                    or abs(int(old_pos) % 15 - pos % 15) == 14):
                    record_first("wrap_move", {"step": step, "uid": uid, "role": role,
                                               "from_cell": int(old_pos), "to_cell": pos})
            if phase and role in raw[phase]:
                c = raw[phase][role]
                c["worker_states"] += 1
                c["enemy_zone_states"] += int(in_enemy)
                c["enemy_near_states"] += int(near_enemy)
                if do is not None:
                    c["own_distance_sum"] += do
                    c["max_own_distance"] = max(c["max_own_distance"], do)
                visits[phase][role].add(pos)
        current_trees = {}
        near_count = zone_count = 0
        for owner, player in enumerate(players):
            for tid, data in player[3].items():
                pos, age = int(data[0]), int(data[1])
                current_trees[pos] = (owner, tid, age)
                if owner != seat:
                    continue
                do, de = _nearest(pos, own_bases), _nearest(pos, enemy_bases)
                near_count += int(de is not None and de <= 2)
                zone_count += int(de is not None and do is not None and de < do)
                old = previous_trees.get(pos)
                kind = "ownership_transfer" if old and old[0] != seat else (
                    "new_tree_observed" if old is None else "already_owned")
                detail = {"step": step, "tree": tid, "cell": pos, "rc": _point(pos),
                          "age": age, "d_own": do, "d_enemy": de, "kind": kind}
                record_first("own_tree", dict(detail))
                if de is not None and de <= 2:
                    record_first("enemy_near_tree", dict(detail))
                if phase and (old is None or old[0] != seat):
                    tree_raw[phase]["new_ownership_events"] += 1
                    tree_raw[phase]["near_enemy_new_ownership_events"] += int(de is not None and de <= 2)
                    tree_raw[phase]["ownership_transfers"] += int(old is not None and old[0] != seat)
        if phase:
            tree_raw[phase]["states"] += 1
            tree_raw[phase]["own_tree_states"] += len(own[3])
            tree_raw[phase]["enemy_near_tree_states"] += near_count
            if near_count > tree_raw[phase]["peak_enemy_near_trees"]:
                tree_raw[phase]["peak_enemy_near_trees"] = near_count
                tree_raw[phase]["peak_enemy_near_step"] = step
        if step in (10, 25, 50, 100, 150):
            role_counts = Counter(v[2] for v in workers.values())
            nearest_base = Counter()
            for pos, _, role in workers.values():
                if role == "COLLECTOR" and own_bases:
                    ds = [distance(pos, b) for b in own_bases]
                    nearest_base["tie" if ds.count(min(ds)) > 1 else str(own_bases[ds.index(min(ds))])] += 1
            checkpoints.append({"step": step, "cash": round(float(own[0]), 2),
                                "collectors": role_counts["COLLECTOR"], "planters": role_counts["PLANTER"],
                                "trees": len(own[3]), "enemy_zone_trees": zone_count,
                                "enemy_near_trees": near_count, "collectors_nearest_own_base": dict(nearest_base)})
        previous_workers = {uid: list(data) for uid, data in workers.items()}
        previous_trees = current_trees

    for phase, role_stats in raw.items():
        phases[phase] = {}
        for role, values in role_stats.items():
            n = values["worker_states"]
            phases[phase][role.lower()] = {
                "worker_states": n,
                "enemy_zone_fraction": round(values["enemy_zone_states"] / n, 4) if n else None,
                "enemy_near_fraction": round(values["enemy_near_states"] / n, 4) if n else None,
                "mean_own_base_distance": round(values["own_distance_sum"] / n, 3) if n else None,
                "max_own_base_distance": values["max_own_distance"] if n else None,
                "unique_cells": len(visits[phase][role])}
        tr = tree_raw[phase]
        phases[phase]["trees"] = {
            "new_ownership_events": tr["new_ownership_events"],
            "near_enemy_new_ownership_events": tr["near_enemy_new_ownership_events"],
            "ownership_transfers": tr["ownership_transfers"],
            "peak_enemy_near_count": tr["peak_enemy_near_trees"],
            "peak_enemy_near_step": tr.get("peak_enemy_near_step"),
            "mean_enemy_near_count": round(tr["enemy_near_tree_states"] / tr["states"], 3) if tr["states"] else None}
        rs = returns[phase]
        phases[phase]["collector_returns"] = {
            "count": len(rs), "mean_cargo_before": round(sum(x["cargo_before"] for x in rs) / len(rs), 2) if rs else None,
            "first_two": rs[:2]}
    events.sort(key=lambda x: (x["step"], x["event"]))
    return {"valid_states": valid_states, "first": first, "opening_recruits": recruits[:6],
            "first_planter": first_planter, "opening_events": events, "phases": phases,
            "checkpoints": checkpoints}


if __name__ == "__main__":
    import argparse
    import gzip
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay")
    parser.add_argument("--seat", type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    opener = gzip.open if args.replay.endswith(".gz") else open
    with opener(args.replay, "rt") as fh:
        replay = json.load(fh)
    print(json.dumps(analyze_space(replay["steps"], args.seat), ensure_ascii=False, indent=2))
