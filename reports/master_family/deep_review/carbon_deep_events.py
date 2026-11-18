"""Public-observation event extraction for Carbon replay snapshots.

Actions on row t lead from observation t-1 to observation t.  Worker cargo,
positions, types, bases and trees are public; full_carbon is never accessed.
Rules: helpers.py Board.next(), recruitment/movement/collision/base-deposit
phases (951-1100).  This is an event audit, not a full game simulator.
"""

from collections import defaultdict
import math


DEFAULT_RULES = {
    "size": 15, "moveCost": 0.0, "smeltCost": 0.0,
    "recCollectorCost": 30.0, "recCollectorIncreaseCost": 0.0,
    "recPlanterCost": 30.0, "recPlanterIncreaseCost": 0.0,
    "workerLimit": 10, "treeLifespan": 50,
}
WINDOWS = ((0, 100), (100, 200), (200, 300))
MOVES = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
NUMERIC_METRICS = (
    "verified_transitions", "skipped_transitions", "delivery_count",
    "delivered_carbon", "delivery_net_cash", "enemy_delivery_to_us_count",
    "enemy_delivery_to_us_net_cash", "delivery_to_enemy_count",
    "delivered_to_enemy_carbon", "collision_count", "contested_collision_count",
    "own_same_side_collision_count", "collision_worker_deaths",
    "collision_collector_deaths", "collision_planter_deaths",
    "collision_removed_own_cargo_gross", "collision_own_cargo_retained",
    "collision_own_cargo_to_opponent", "collision_own_cargo_to_ground",
    "collision_enemy_cargo_captured", "enemy_base_elimination_count",
    "recruits", "trees_planted", "trees_captured", "trees_lost_to_capture",
    "trees_expired",
)


def _number(value, label, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + " must be numeric")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(label + " must be finite and nonnegative")
    return value


def _integer(value, label):
    n = _number(value, label)
    if int(n) != n:
        raise ValueError(label + " must be integral")
    return int(n)


def _snapshot(row, size):
    if not isinstance(row, (list, tuple)) or len(row) != 2:
        raise ValueError("expected a two-seat replay row")
    observations = [r.get("observation", {}) for r in row if isinstance(r, dict)]
    full = [o for o in observations if "players" in o and "step" in o]
    if not full:
        raise ValueError("missing shared public observation")
    obs = full[0]
    if any(o["players"] != obs["players"] or o["step"] != obs["step"] for o in full[1:]):
        raise ValueError("inconsistent shared observations")
    step = _integer(obs["step"], "step")
    if len(obs["players"]) != 2:
        raise ValueError("expected two players")
    workers, trees, bases, cash = {}, {}, {}, []
    for owner, player in enumerate(obs["players"]):
        if not isinstance(player, (list, tuple)) or len(player) != 4:
            raise ValueError("invalid public player record")
        cash.append(_number(player[0], "cash"))
        for bid, pos in player[1].items():
            pos = _integer(pos, "base position")
            if pos >= size * size or pos in bases:
                raise ValueError("invalid or duplicate base position")
            bases[pos] = {"id": str(bid), "seat": owner}
        for wid, values in player[2].items():
            if len(values) != 3 or str(wid) in workers:
                raise ValueError("invalid or duplicate worker")
            pos = _integer(values[0], "worker position")
            if pos >= size * size or values[2] not in ("COLLECTOR", "PLANTER"):
                raise ValueError("invalid worker position/type")
            workers[str(wid)] = {"id": str(wid), "seat": owner, "pos": pos,
                                 "cargo": _number(values[1], "cargo"),
                                 "type": values[2], "new": False}
        for tid, values in player[3].items():
            if len(values) != 2 or str(tid) in trees:
                raise ValueError("invalid or duplicate tree")
            pos = _integer(values[0], "tree position")
            if pos >= size * size:
                raise ValueError("invalid tree position")
            trees[str(tid)] = {"id": str(tid), "seat": owner, "pos": pos,
                               "age": _integer(values[1], "tree age")}
    if len({w["pos"] for w in workers.values()}) != len(workers):
        raise ValueError("multiple workers in a post-transition cell")
    return {"step": step, "workers": workers, "trees": trees, "bases": bases, "cash": cash}


def _actions(row):
    result = []
    for r in row:
        if "action" not in r:
            raise ValueError("missing recorded transition action")
        a = r["action"] or {}
        if not isinstance(a, dict):
            raise ValueError("invalid action mapping")
        result.append(a)
    return result


def _reconstruct(before, after, actions, rules, seat):
    size = rules["size"]
    live = {wid: dict(w) for wid, w in before["workers"].items()}
    events = []

    def event(kind, pos, **fields):
        out = {"kind": kind, "step_from": before["step"], "step_to": after["step"],
               "position": pos, "row": pos // size, "col": pos % size,
               "evidence": "public_rules_reconstruction_verified_against_next_roster"}
        out.update(fields)
        events.append(out)
        return out

    # Recruitment prices use global counts at the beginning of this transition.
    prices = {}
    for typ, suffix in (("COLLECTOR", "Collector"), ("PLANTER", "Planter")):
        prices[typ] = (rules["rec" + suffix + "Cost"] +
                       sum(w["type"] == typ for w in live.values()) *
                       rules["rec" + suffix + "IncreaseCost"])
    for owner in (0, 1):
        cash = before["cash"][owner]
        count = sum(w["seat"] == owner for w in live.values())
        for pos, base in before["bases"].items():
            if base["seat"] != owner:
                continue
            act = actions[owner].get(base["id"])
            typ = {"RECCOLLECTOR": "COLLECTOR", "RECPLANTER": "PLANTER"}.get(act)
            if typ and cash >= prices[typ] and count < rules["workerLimit"]:
                cash -= prices[typ]
                count += 1
                wid = "__new__%d_%s_%d" % (owner, base["id"], before["step"])
                live[wid] = {"id": wid, "seat": owner, "pos": pos, "cargo": 0.0,
                             "type": typ, "new": True}
                if owner == seat:
                    event("recruit", pos, worker_id=wid, worker_type=typ,
                          cost=prices[typ], known_amount=0.0)
        for w in live.values():
            if w["seat"] != owner or w["new"]:
                continue
            action = actions[owner].get(w["id"])
            if action in MOVES:
                dr, dc = MOVES[action]
                r, c = divmod(w["pos"], size)
                w["pos"] = ((r + dr) % size) * size + (c + dc) % size
                w["cargo"] *= 1.0 - rules["moveCost"]
    surviving_trees = {t["pos"]: t for t in before["trees"].values()
                       if t["age"] < rules["treeLifespan"]}
    groups = defaultdict(list)
    for w in live.values():
        groups[w["pos"]].append(w)
    for pos, workers in groups.items():
        if len(workers) < 2:
            continue
        collectors = [w for w in workers if w["type"] == "COLLECTOR"]
        smallest = min((w["cargo"] for w in collectors), default=None)
        winners = [w for w in collectors if w["cargo"] == smallest]
        winner = winners[0] if len(winners) == 1 else None
        deleted = [w for w in workers if w is not winner]
        if winner is not None:
            recipient, recipient_seat = "collector", winner["seat"]
        elif pos in surviving_trees:
            recipient, recipient_seat = "tree_cash", surviving_trees[pos]["seat"]
        elif pos in before["bases"]:
            recipient, recipient_seat = "base_cash", before["bases"][pos]["seat"]
        else:
            recipient, recipient_seat = "ground", None
        ours = [w for w in deleted if w["seat"] == seat]
        own_cargo = sum(w["cargo"] for w in ours)
        enemy_cargo = sum(w["cargo"] for w in deleted if w["seat"] != seat)
        if any(w["seat"] == seat for w in workers) or recipient_seat == seat:
            event("collision", pos,
                  participants=[{"worker_id": w["id"], "seat": w["seat"],
                                 "type": w["type"], "known_cargo": w["cargo"]}
                                for w in workers],
                  deleted_worker_ids=[w["id"] for w in deleted],
                  winner_worker_id=winner["id"] if winner else None,
                  recipient=recipient, recipient_seat=recipient_seat,
                  contested=len({w["seat"] for w in workers}) > 1,
                  own_participants=sum(w["seat"] == seat for w in workers),
                  enemy_participants=sum(w["seat"] != seat for w in workers),
                  own_deaths=len(ours),
                  own_collector_deaths=sum(w["type"] == "COLLECTOR" for w in ours),
                  own_planter_deaths=sum(w["type"] == "PLANTER" for w in ours),
                  own_removed_cargo_gross=own_cargo,
                  own_cargo_retained=own_cargo if recipient_seat == seat else 0.0,
                  own_cargo_to_opponent=own_cargo if recipient_seat == 1 - seat else 0.0,
                  own_cargo_to_ground=own_cargo if recipient_seat is None else 0.0,
                  enemy_cargo_captured=enemy_cargo if recipient_seat == seat else 0.0,
                  known_amount=own_cargo + (enemy_cargo if recipient_seat == seat else 0.0))
        if winner:
            winner["cargo"] += sum(w["cargo"] for w in deleted)
        for w in deleted:
            del live[w["id"]]
    for w in list(live.values()):
        base = before["bases"].get(w["pos"])
        if not base or w["type"] != "COLLECTOR":
            continue
        if w["seat"] != base["seat"] and w["cargo"] <= rules["smeltCost"]:
            if w["seat"] == seat:
                event("enemy_base_elimination", w["pos"], worker_id=w["id"],
                      known_amount=w["cargo"], known_cargo=w["cargo"])
            del live[w["id"]]
        elif w["cargo"] >= rules["smeltCost"]:
            cargo, cash = w["cargo"], w["cargo"] - rules["smeltCost"]
            if cargo > 0 and (w["seat"] == seat or base["seat"] == seat):
                kind = ("delivery" if w["seat"] == seat == base["seat"] else
                        "delivery_to_enemy" if w["seat"] == seat else "enemy_delivery_to_us")
                event(kind, w["pos"], worker_id=w["id"], carrier_seat=w["seat"],
                      base_seat=base["seat"], gross_carbon=cargo,
                      net_cash=cash, known_amount=cargo)
            w["cargo"] = 0.0
            w["deposited"] = True

    # Verify identities/positions/types, including newborn IDs.  Collection and
    # tree absorption happen later, so general post-transition cargo is NOT tested.
    actual = after["workers"]
    unmatched = set(actual) - set(before["workers"])
    aliases = {}
    predicted_ids = set()
    for wid, w in live.items():
        actual_id = wid
        if w["new"]:
            matches = [nid for nid in unmatched if all(actual[nid][k] == w[k]
                       for k in ("seat", "pos", "type"))]
            if len(matches) != 1:
                raise ValueError("newborn survivor does not match next public roster")
            actual_id = matches[0]
            unmatched.remove(actual_id)
            aliases[wid] = actual_id
        if actual_id not in actual or any(actual[actual_id][k] != w[k]
                                          for k in ("seat", "pos", "type")):
            raise ValueError("predicted survivor does not match next public roster: " + wid)
        if w.get("deposited") and abs(actual[actual_id]["cargo"]) > 1e-8:
            raise ValueError("post-deposit cargo is not zero")
        predicted_ids.add(actual_id)
    if predicted_ids != set(actual):
        raise ValueError("predicted deaths/new recruits do not match next public roster")
    if before["bases"] != after["bases"]:
        raise ValueError("bases changed across transition")
    # Replace surviving symbolic recruit IDs everywhere; dead newborn IDs remain
    # explicitly synthetic because the engine never exposed them in a snapshot.
    def replace_ids(obj):
        if isinstance(obj, dict):
            return {k: replace_ids(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [replace_ids(v) for v in obj]
        return aliases.get(obj, obj) if isinstance(obj, str) else obj
    events = replace_ids(events)
    for tid, tree in after["trees"].items():
        old = before["trees"].get(tid)
        if old is None and tree["seat"] == seat:
            event("tree_planted", tree["pos"], tree_id=tid, known_amount=0.0,
                  evidence="observed_tree_identity_change")
        elif old and old["seat"] != tree["seat"]:
            event("tree_captured" if tree["seat"] == seat else "tree_lost_to_capture",
                  tree["pos"], tree_id=tid, known_amount=0.0,
                  evidence="observed_tree_owner_change")
    for tid, tree in before["trees"].items():
        if tid not in after["trees"] and tree["seat"] == seat:
            if tree["age"] >= rules["treeLifespan"]:
                event("tree_expired", tree["pos"], tree_id=tid, known_amount=0.0,
                      evidence="observed_disappearance_at_rule_lifespan")
            else:
                raise ValueError("unexplained disappearance of non-expired tree")
    return events


def _empty_metrics():
    return {k: 0 for k in NUMERIC_METRICS}


def _add_event(summary, e):
    kind = e["kind"]
    if kind == "delivery":
        summary["delivery_count"] += 1
        summary["delivered_carbon"] += e["gross_carbon"]
        summary["delivery_net_cash"] += e["net_cash"]
    elif kind == "enemy_delivery_to_us":
        summary["enemy_delivery_to_us_count"] += 1
        summary["enemy_delivery_to_us_net_cash"] += e["net_cash"]
    elif kind == "delivery_to_enemy":
        summary["delivery_to_enemy_count"] += 1
        summary["delivered_to_enemy_carbon"] += e["gross_carbon"]
    elif kind == "collision":
        summary["collision_count"] += 1
        summary["contested_collision_count"] += int(e["contested"])
        summary["own_same_side_collision_count"] += int(e["own_participants"] > 1 and not e["contested"])
        for src, dst in (("own_deaths", "worker_deaths"),
                         ("own_collector_deaths", "collector_deaths"),
                         ("own_planter_deaths", "planter_deaths"),
                         ("own_removed_cargo_gross", "removed_own_cargo_gross"),
                         ("own_cargo_retained", "own_cargo_retained"),
                         ("own_cargo_to_opponent", "own_cargo_to_opponent"),
                         ("own_cargo_to_ground", "own_cargo_to_ground"),
                         ("enemy_cargo_captured", "enemy_cargo_captured")):
            summary["collision_" + dst] += e[src]
    elif kind == "enemy_base_elimination":
        summary["enemy_base_elimination_count"] += 1
    else:
        key = {"recruit": "recruits", "tree_planted": "trees_planted",
               "tree_captured": "trees_captured", "tree_lost_to_capture": "trees_lost_to_capture",
               "tree_expired": "trees_expired"}.get(kind)
        if key:
            summary[key] += 1


def analyze_events(steps, seat, *, configuration=None):
    """Return JSON-native whole-game/window events, excluding bad transitions.

    Pass replay['configuration'] for rule-bound conclusions.  With two arguments
    alone, the default competition rules are explicitly labelled assumed.  Step
    windows refer to the decision/source step, NOT the destination replay row.
    Every inferred transition is atomic: any verification error drops all of its
    events and records an anomaly, never converting unexplained losses into facts.
    """
    if isinstance(seat, bool) or seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    config = configuration or {}
    rules = dict(DEFAULT_RULES)
    assumed = [k for k in rules if k not in config]
    rules.update({k: config[k] for k in rules if k in config})
    for k in rules:
        rules[k] = _integer(rules[k], k) if k in ("size", "workerLimit", "treeLifespan") else _number(rules[k], k)
    if rules["size"] < 1 or rules["moveCost"] > 1:
        raise ValueError("invalid map size or moveCost")
    summary = _empty_metrics()
    windows = [{"step_start": lo, "step_end_exclusive": hi, "metrics": _empty_metrics()}
               for lo, hi in WINDOWS]
    events, anomalies = [], []
    for i in range(1, len(steps)):
        source_step = None
        window = None
        try:
            before = _snapshot(steps[i - 1], rules["size"])
            source_step = before["step"]
            window = next((w["metrics"] for w in windows
                           if w["step_start"] <= source_step < w["step_end_exclusive"]), None)
            after = _snapshot(steps[i], rules["size"])
            if after["step"] != source_step + 1:
                raise ValueError("nonconsecutive observation steps")
            if any(r.get("status") not in ("ACTIVE", "DONE")
                   for row in (steps[i - 1], steps[i]) for r in row):
                raise ValueError("missing or unhealthy agent status")
            batch = _reconstruct(before, after, _actions(steps[i]), rules, seat)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            summary["skipped_transitions"] += 1
            if window is not None:
                window["skipped_transitions"] += 1
            anomalies.append({"destination_row": i, "step_from": source_step,
                              "reason": str(exc), "events_counted": False})
            continue
        summary["verified_transitions"] += 1
        if window is not None:
            window["verified_transitions"] += 1
        for e in batch:
            e["destination_row"] = i
            _add_event(summary, e)
            if window is not None:
                _add_event(window, e)
        events.extend(batch)
    material = [e for e in events if e["known_amount"] > 0]
    largest = sorted(material, key=lambda e: (-e["known_amount"], e["step_from"], e["position"]))[:2]
    losses = [e for e in events if e["kind"] == "collision" and
              e["own_cargo_to_opponent"] + e["own_cargo_to_ground"] > 0]
    losses = sorted(losses, key=lambda e: -(e["own_cargo_to_opponent"] + e["own_cargo_to_ground"]))[:2]
    return {"schema_version": "carbon-public-events-v1", "seat": int(seat),
            "rules": rules, "assumed_rule_keys": assumed,
            "rule_binding": "replay_configuration" if not assumed else "contains_assumed_defaults",
            "window_basis": "source decision step: [0,100), [100,200), [200,300)",
            "metrics": summary, "windows": windows, "events": events,
            "largest_events": largest, "largest_collision_losses": losses,
            "anomalies": anomalies,
            "limitations": [
                "Only public players/step/action/status fields are read; carbon/full_carbon are not used.",
                "Rule reconstruction is conditional on the bound rule version/configuration; roster checks are consistency tests, not proof of every hidden rule.",
                "Cargo drop alone is not a delivery or collision loss: tree absorption and other phases can change cargo.",
                "Collision gross removed cargo includes same-side transfers; use opponent/ground buckets for lost custody.",
                "Ground spill is cargo removed from workers; retained ground carbon after the map cap is not observable here.",
                "Collision capture and subsequent base delivery may describe the same carbon; event totals are not additive final-score attribution.",
                "net_cash is the transfer before the engine's final per-turn cash rounding; cash changes also include recruitment, planting, capture costs and tree absorption.",
                "largest_events ranks known material amount, not tactical merit, avoidability, or counterfactual score impact.",
                "Worker disappearance without a fully verified phase reconstruction is excluded and listed as an anomaly.",
                "Newborn workers killed in their recruitment transition have synthetic __new__ IDs; survivor IDs are matched to the next snapshot.",
            ]}
