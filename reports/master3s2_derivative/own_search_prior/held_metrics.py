"""Score the actual native A0 policy against historical final orders, with common private samples."""

from collections import Counter, defaultdict
import types

import numpy as np

STEPS = tuple(range(7, 294, 13))


def canonical(order):
    return {uid: action for uid, action in order.items() if action and action != "NONE"}


def native_a0(model, obs, x, lw, lb, seed):
    """Reuse the source sampler, go_home guard, base spending and joint occupancy in native _act."""
    if not np.array_equal(model._quant(x), x):
        raise ValueError("Expected the source policy's already rounded features")
    tracker = types.SimpleNamespace(step=lambda *args: (None, None), set_actions=lambda *args: None)
    ns = dict(
        model.__dict__,
        _ST=dict(tr=tracker, prev=None, rng=np.random.RandomState(seed)),
        _POLICY=types.SimpleNamespace(forward=lambda *args: (lw, lb)),
        encode=lambda *args: x,
    )
    decode = types.FunctionType(model._act.__code__, ns, argdefs=model._act.__defaults__)
    return canonical(decode(obs, None))


def nll(logits, index):
    z = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite native prior logits")
    return float(np.logaddexp.reduce(z) - z[index])


def add_hits(counter, prefix, correct, old, new):
    counter[prefix + "_n"] += 1
    counter[prefix + "_old_hit"] += int(old == correct)
    counter[prefix + "_new_hit"] += int(new == correct)
    if old != new:
        counter[prefix + "_changed"] += 1
        if new == correct:
            counter[prefix + "_fixed"] += 1
        elif old == correct:
            counter[prefix + "_harmed"] += 1
        else:
            counter[prefix + "_both_wrong"] += 1


def event(counter, name, actual, old, new):
    counter[name + "_opportunities"] += 1
    counter[name + "_actual"] += int(actual)
    for who, forecast in (("old", old), ("new", new)):
        outcome = ("tp" if forecast else "fn") if actual else ("fp" if forecast else "tn")
        counter[f"{name}_{who}_{outcome}"] += 1
    return dict(actual=actual, old=old, new=new)


def score_root(row, game, model, weights, correct):
    obs = row["obs"]
    step, me = row["step"], game["seat"]
    if obs["player"] != me or obs["step"] != step or "full_carbon" in obs:
        raise ValueError("Invalid expert seat before issuing its move")
    seed = 9023 + 1000003 * (step + 1) + 17 * me + 31 * (game["seed"] % 10000)
    lw, lb = np.asarray(row["lw"]), np.asarray(row["lb"])
    nw, nb = correct(weights, row["h"], lw.copy(), lb.copy())
    old = native_a0(model, obs, row["x"], lw, lb, seed)
    new = native_a0(model, obs, row["x"], nw, nb, seed)

    # The predictor sees neither simultaneous command. They now become labels.
    own, other = canonical(row["own"]), canonical(row["other"])
    players = obs["players"]
    cash, own_bases, units, _ = players[me]
    enemy = players[1 - me]
    bases = set(own_bases.values())
    enemy_trees = {tree[0] for tree in enemy[3].values()}
    enemy_collectors = [
        (w[0], w[1], model.move(w[0], model.WIDX[other.get(uid, "NONE")]))
        for uid, w in enemy[2].items()
        if w[2] == "COLLECTOR"
    ]
    stats = Counter()
    add_hits(stats, "joint", own, old, new)
    cases = []
    for uid, (pos, load, role) in units.items():
        y, a, b = (d.get(uid, "NONE") for d in (own, old, new))
        add_hits(stats, "worker", y, a, b)
        stats["worker_old_nll_sum"] += nll(lw[pos], model.WIDX[y])
        stats["worker_new_nll_sum"] += nll(nw[pos], model.WIDX[y])
        loaded = role == "COLLECTOR" and load >= 50
        nearest_home = min(model.dist(pos, bp) for bp in bases)
        rivals = (
            [
                (place, dest)
                for place, mass, dest in enemy_collectors
                if mass < load and model.dist(pos, place) <= 2
            ]
            if loaded
            else []
        )
        deposit = loaded and nearest_home == 1
        capture = role == "PLANTER" and pos in enemy_trees
        if not (deposit or rivals or capture):
            continue
        add_hits(stats, "economic", y, a, b)
        stats["economic_old_nll_sum"] += nll(lw[pos], model.WIDX[y])
        stats["economic_new_nll_sum"] += nll(nw[pos], model.WIDX[y])
        destinations = [model.move(pos, model.WIDX[v]) for v in (y, a, b)]
        events = {}
        if deposit:
            events["deposit"] = event(stats, "deposit", *(d in bases for d in destinations))
        if rivals:
            rival_destinations = {d for _, d in rivals}
            events["lighter_contact"] = event(
                stats, "lighter_contact", *(d in rival_destinations for d in destinations)
            )
        if capture:
            events["capture_stay"] = event(stats, "capture_stay", *(d == pos for d in destinations))
        cases.append(
            dict(uid=uid, cell=pos, cargo=load, role=role, actual=y, old=a, new=b, events=events)
        )
    stats["roots_with_economic_opportunity"] += bool(cases)
    affordable = len(units) < 10 and cash >= 30
    for uid, pos in own_bases.items():
        y, a, b = (d.get(uid, "NONE") for d in (own, old, new))
        for key in (["base", "base_before_joint_available"] if affordable else ["base"]):
            add_hits(stats, key, y, a, b)
            stats[key + "_old_nll_sum"] += nll(lb[pos], model.BIDX[y])
            stats[key + "_new_nll_sum"] += nll(nb[pos], model.BIDX[y])
    return stats, dict(
        terrain=game["terrain"],
        seed=game["seed"],
        seat=me,
        step=step,
        rng_seed=seed,
        actual=own,
        enemy_actual=other,
        old=old,
        new=new,
        economic_cases=cases,
    )


def readable(stats):
    out = dict(stats)
    for unit in ("joint", "worker", "economic", "base", "base_before_joint_available"):
        for k in ("n", "old_hit", "new_hit", "changed", "fixed", "harmed", "both_wrong"):
            out.setdefault(unit + "_" + k, 0)
        if (
            out[unit + "_new_hit"] - out[unit + "_old_hit"]
            != out[unit + "_fixed"] - out[unit + "_harmed"]
        ):
            raise ValueError("Unreconciled accuracy change")
        if unit != "joint":
            for who in ("old", "new"):
                out[unit + "_" + who + "_mean_nll"] = (
                    stats[unit + "_" + who + "_nll_sum"] / stats[unit + "_n"]
                    if stats[unit + "_n"]
                    else None
                )
    for name in ("deposit", "lighter_contact", "capture_stay"):
        for cond in (
            "opportunities",
            "actual",
            *(a + "_" + b for a in ("old", "new") for b in ("tp", "fp", "fn", "tn")),
        ):
            out.setdefault(name + "_" + cond, 0)
        for who in ("old", "new"):
            if out[f"{name}_{who}_tp"] + out[f"{name}_{who}_fn"] != out[name + "_actual"]:
                raise ValueError("Unreconciled positive events")
            if (
                sum(out[f"{name}_{who}_{a}"] for a in ("tp", "fp", "fn", "tn"))
                != out[name + "_opportunities"]
            ):
                raise ValueError("Unreconciled event opportunities")
    return out


def grade(games, model, weights, correct):
    """Evaluate one four-map panel (old held or completed dev1 games) and its gate independently."""
    roster = [(g["terrain"], g["seat"]) for g in games]
    if len(roster) != len(set(roster)) or len(set(t for t, _ in roster)) != 4:
        raise ValueError("Expected one four-map held panel; duplicate map/seat")
    by_map = defaultdict(Counter)
    totals = Counter()
    rows = []
    map_seed = {}
    for g in games:
        terrain = g["terrain"]
        if terrain in map_seed and map_seed[terrain] != g["seed"]:
            raise ValueError("Terrain aliases in held panel")
        map_seed[terrain] = g["seed"]
        if [r["step"] for r in g["rows"]] != list(STEPS) or g["seed"] in (621758590, 621758591):
            raise ValueError("Wrong archive or registered root list")
        for row in g["rows"]:
            counted, detail = score_root(row, g, model, weights, correct)
            by_map[terrain].update(counted)
            totals.update(counted)
            rows.append(detail)
    n_worker = sum(st["worker_new_hit"] >= st["worker_old_hit"] for st in by_map.values())
    n_joint = sum(st["joint_new_hit"] >= st["joint_old_hit"] for st in by_map.values())
    checks = dict(
        worker_hits_strict=totals["worker_new_hit"] > totals["worker_old_hit"],
        joint_hits_nonnegative=totals["joint_new_hit"] >= totals["joint_old_hit"],
        economic_hits_strict=totals["economic_new_hit"] > totals["economic_old_hit"],
        worker_nonnegative_3_of_4=(n_worker >= 3),
        joint_nonnegative_3_of_4=(n_joint >= 3),
    )
    return dict(
        pass_gate=all(checks.values()),
        checks=checks,
        counts=readable(totals),
        nonnegative_maps=dict(worker=n_worker, joint=n_joint),
        by_map={k: dict(seed=map_seed[k], **readable(st)) for k, st in sorted(by_map.items())},
        roots=rows,
        definitions=dict(
            root="actual original post-search expert team order versus source protected A0 with common RNG",
            NLL="raw action-head negative log probability in nats before joint sampling and guards",
            deposit="loaded >=50 collector initially one step from own base, issued destination equals own base",
            lighter_contact="loaded >=50 collector, strictly lighter enemy collector initially within two, projected simultaneous destination equality",
            capture_stay="own planter standing on enemy tree and issuing stay; before referee collision, available funds, and capture settlement",
            economic_union="worker counted once per root across deposit, lighter contact, and capture opportunities",
            hit="exact emitted action; event TP/FP/FN/TN compare projected pre-resolution event with actual issued orders",
        ),
    )
