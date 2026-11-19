"""Descriptive next-transition metrics for registered same-worker stay/move pairs."""

from collections import Counter
from math import fsum, isfinite
from numbers import Integral, Real


def boolean(value, field):
    if hasattr(value, "item"):
        value = value.item()
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def number(value, field, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a number")
    v = float(value)
    if not isfinite(v) or v < 0 or (maximum is not None and v > maximum):
        raise ValueError(f"{field} outside its finite range")
    return v


def checked(rows):
    seen = set()
    clean = []
    for r in rows:
        keys = (r["seed"], r["seat"], r["t"])
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in keys):
            raise ValueError("seed, physical seat and turn must be nonnegative integers")
        seed, seat, t = map(int, keys)
        uid = r["uid"]
        if seat not in (0, 1) or isinstance(uid, bool) or not isinstance(uid, (str, Integral)) or uid == "":
            raise ValueError("invalid seat or worker ID")
        key = (seed, seat, t, uid)
        if key in seen:
            raise ValueError(f"repeated root {key}")
        seen.add(key)
        actions = r["actions"]
        if (
            len(actions) != 2
            or actions[0]["name"] != "NONE"
            or actions[1]["name"] not in ("UP", "RIGHT", "DOWN", "LEFT")
        ):
            raise ValueError("expected registered stationary action then a physical movement")
        arms = []
        for a in actions:
            arms.append(
                dict(
                    name=a["name"],
                    died=boolean(a["died"], "died"),
                    paid=number(a["paid"], "paid"),
                    p=number(a["p"], "p", 1),
                    geometric=boolean(a["geometric"], "geometric"),
                )
            )
        if arms[1]["paid"] or (arms[0]["died"] and arms[0]["paid"]):
            raise ValueError("the focal-worker acquisition must be surviving and stationary")
        price = None if r["physical_site"] is None else number(r["physical_site"], "price")
        clean.append(
            dict(
                seed=seed,
                seat=seat,
                t=t,
                uid=uid,
                price=price,
                payable=boolean(r["payable"], "can_pay_now"),
                arms=arms,
            )
        )
    if not clean:
        raise ValueError("no paired observations")
    return clean


def ratio(a, b):
    return a / b if b else None


def counts(rows):
    return {
        "maps": len({r["seed"] for r in rows}),
        "map_seats_with_roots": len({(r["seed"], r["seat"]) for r in rows}),
        "worker_instances": len({(r["seed"], r["seat"], r["uid"]) for r in rows}),
        "pairs": len(rows),
        "arms": 2 * len(rows),
    }


def exposure(rows):
    stay = [r["arms"][0] for r in rows]
    payments = [a["paid"] for a in stay if a["paid"] > 0]
    return {
        **counts(rows),
        "price_quoted": sum(r["price"] is not None for r in rows),
        "quoted_without_can_pay_now": sum(r["price"] is not None and not r["payable"] for r in rows),
        "can_pay_now": sum(r["payable"] for r in rows),
        "actually_paid_on_stay": len(payments),
        "actual_stay_spend": fsum(payments),
        "can_pay_now_and_stay_died": sum(r["payable"] and a["died"] for r, a in zip(rows, stay)),
        "can_pay_now_and_stay_alive_unpaid": sum(
            r["payable"] and not a["died"] and not a["paid"] for r, a in zip(rows, stay)
        ),
        "can_pay_now_and_stay_paid": sum(r["payable"] and a["paid"] > 0 for r, a in zip(rows, stay)),
        "without_can_pay_now_and_stay_paid": sum(
            not r["payable"] and a["paid"] > 0 for r, a in zip(rows, stay)
        ),
    }


def alert_score(weighted, rule, arms):
    table = Counter()
    masses = Counter()
    for row, weight in weighted:
        for k in arms:
            a = row["arms"][k]
            category = ("tp" if rule(a) else "fn") if a["died"] else ("fp" if rule(a) else "tn")
            table[category] += 1
            masses[category] += weight / len(arms)
    return {
        "counts": {c: table[c] for c in ("tp", "fn", "fp", "tn")},
        "arms": sum(table.values()),
        "actually_dead": table["tp"] + table["fn"],
        "actually_alive": table["tn"] + table["fp"],
        "alerted": table["tp"] + table["fp"],
        "precision": ratio(masses["tp"], masses["tp"] + masses["fp"]),
        "recall": ratio(masses["tp"], masses["tp"] + masses["fn"]),
        "false_positive_rate": ratio(masses["fp"], masses["tn"] + masses["fp"]),
        "false_negative_rate": ratio(masses["fn"], masses["tp"] + masses["fn"]),
    }


def policy_score(weighted, choose_move):
    moved = saved = lost = paid_forgone = kept_paid = 0
    delta_mass = chosen_death = chosen_paid = foregone_mass = foregone_cash = cash_mass = 0.0
    for row, w in weighted:
        stay, move = row["arms"]
        switch = choose_move(stay, move)
        moved += switch
        selected = move if switch else stay
        delta = int(stay["died"]) - int(selected["died"])
        delta_mass += w * delta
        chosen_death += w * selected["died"]
        chosen_paid += w * (selected["paid"] > 0)
        saved += delta > 0
        lost += delta < 0
        kept_paid += selected["paid"] > 0
        foregone = switch and stay["paid"] > 0
        paid_forgone += foregone
        foregone_mass += w * foregone
        foregone_cash += stay["paid"] if foregone else 0
        cash_mass += w * stay["paid"] if foregone else 0
    return {
        "pairs": len(weighted),
        "selected_move": moved,
        "selected_stay": len(weighted) - moved,
        "saved_by_moving": saved,
        "deaths_caused_by_moving": lost,
        "net_saved_vs_always_stay": saved - lost,
        "net_saved_per_root_vs_always_stay": delta_mass,
        "chosen_death_rate": chosen_death,
        "chosen_purchases": kept_paid,
        "chosen_purchase_rate": chosen_paid,
        "foregone_actual_stay_purchases": paid_forgone,
        "foregone_purchases_per_root": foregone_mass,
        "foregone_purchase_debit": foregone_cash,
        "foregone_debit_per_root": cash_mass,
    }


def distribution(rows):
    per_map = Counter(r["seed"] for r in rows)
    if not per_map:
        return {"counts": exposure([]), "evaluated_maps": [], "metrics": None}
    weighted = [(r, 1 / (len(per_map) * per_map[r["seed"]])) for r in rows]
    alert_rules = {
        "fitted_p20": lambda a: a["p"] >= 0.2,
        "fitted_p50": lambda a: a["p"] >= 0.5,
        "geometric": lambda a: a["geometric"],
    }
    alerts = {}
    for label, rule in alert_rules.items():
        values = {
            name: alert_score(weighted, rule, arm)
            for name, arm in (("both_arms", (0, 1)), ("stay", (0,)), ("move", (1,)))
        }
        fp = [any(rule(a) and not a["died"] for a in r["arms"]) for r in rows]
        fn = [any(not rule(a) and a["died"] for a in r["arms"]) for r in rows]
        values["paired_roots"] = {
            "pairs": len(rows),
            "at_least_one_fp": sum(fp),
            "at_least_one_fn": sum(fn),
            "both_fp_and_fn": sum(p and n for p, n in zip(fp, fn)),
        }
        alerts[label] = values
    policy = {
        "always_stay": policy_score(weighted, lambda s, m: False),
        "fitted_p20_delta05": policy_score(weighted, lambda s, m: s["p"] >= 0.2 and s["p"] >= m["p"] + 0.05),
        "geometric": policy_score(weighted, lambda s, m: s["geometric"] and not m["geometric"]),
    }
    outcomes = {}
    for name, death in (
        ("both_survive", (False, False)),
        ("stay_dies_move_survives", (True, False)),
        ("stay_survives_move_dies", (False, True)),
        ("both_die", (True, True)),
    ):
        matching = [w for r, w in weighted if tuple(a["died"] for a in r["arms"]) == death]
        outcomes[name] = {"roots": len(matching), "rate": fsum(matching)}
    brier = {
        model: fsum(w * fsum((a[field] - int(a["died"])) ** 2 for a in r["arms"]) / 2 for r, w in weighted)
        for model, field in (("fitted", "p"), ("binary_geometric", "geometric"))
    }
    return {
        "counts": exposure(rows),
        "evaluated_maps": sorted(per_map),
        "metrics": {"brier": brier, "alerts": alerts, "paired_deaths": outcomes, "policy": policy},
    }


def evaluate(rows):
    """No fitting. Each represented map, then root in that map, then arm share equal weight."""
    data = checked(rows)
    maps = sorted({r["seed"] for r in data})

    def group(subset):
        return {"all": distribution(subset), "can_pay_now": distribution([r for r in subset if r["payable"]])}

    return {
        "schema": "planter-one-step-survival-evaluation/1",
        "weighting": "Counts are physical observations. Rates give equal mass to each represented terrain, then root; all-arm metrics split root mass equally. Precision/recall are computed from these weighted confusion masses. Empty strata and undefined conditional rates are null.",
        "worker_identity": "(source map seed, physical seat, actual worker UID)",
        "payment": "Paid is a verified purchase by this focal worker on this transition. Foregone debit prices missed purchases; their eventual income and terminal effect are unobserved.",
        "counts": exposure(data),
        "maps": {str(s): group([r for r in data if r["seed"] == s]) for s in maps},
        "by_physical_seat": {str(p): group([r for r in data if r["seat"] == p]) for p in (0, 1)},
        "by_map_and_seat": {
            str(s): {str(p): exposure([r for r in data if (r["seed"], r["seat"]) == (s, p)]) for p in (0, 1)}
            for s in maps
        },
        "equal_map": group(data),
    }
