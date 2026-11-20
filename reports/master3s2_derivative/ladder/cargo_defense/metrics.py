"""One-step enemy-collision prediction and decision diagnostics on all five legal worker moves."""
from collections import defaultdict
import math


EPS = 1e-12
ACTIONS = range(5)


def _normalize(row):
    if row["action"] not in ACTIONS or row["seat"] not in (0, 1) or not math.isfinite(row["load"]) or row["load"] < 0:
        raise ValueError("Invalid collector or movement identity")
    arrays = ("risk", "baseline", "labels", "loss", "friend", "home", "occupancy")
    if any(len(row[k]) != 5 or not all(math.isfinite(x) for x in row[k]) for k in arrays):
        raise ValueError("Five finite values required for each legal-action field")
    if any(not 0 <= v <= 1 for k in ("risk", "baseline") for v in row[k]):
        raise ValueError("Invalid probability")
    if row["alive_real"] not in (0, 1) or any(v not in (0, 1) for k in ("labels", "occupancy") for v in row[k]):
        raise ValueError("Non-binary outcome")
    if any(v < 0 for k in ("loss", "friend", "home") for v in row[k]):
        raise ValueError("Negative capture, friendly reach or distance")
    normalized = dict(row)
    for field in arrays:
        normalized[field] = [float(v) for v in row[field]]
    normalized.update(seat=int(row["seat"]), action=int(row["action"]), load=float(row["load"]),
                      alive_real=bool(row["alive_real"]))
    return normalized


def _finish_predict(totals, models):
    for name, m in models.items():
        n = totals["actions"]
        m["nll"] = m.pop("sum_nll") / n if n else None
        m["brier"] = m.pop("sum_brier") / n if n else None
        m["mean_root_auc"] = m.pop("sum_root_auc") / totals["rank_roots"] if totals["rank_roots"] else None
        m["pair_weighted_auc"] = m.pop("sum_auc_credit") / totals["positive_negative_pairs"] if totals["positive_negative_pairs"] else None
    change = {}
    for k in ("nll", "brier", "mean_root_auc", "pair_weighted_auc"):
        change[k] = (None if models["risk"][k] is None else models["risk"][k] - models["baseline"][k])
    totals["models"] = models
    totals["risk_minus_baseline"] = change
    return totals


def _predict(rows, geometric):
    totals = dict(eligible_worker_turns=len(rows), covered_worker_turns=0, actions=0, positive_labels=0,
                  rank_roots=0, positive_negative_pairs=0, root_rank_improved=0, root_rank_worsened=0,
                  root_rank_unchanged=0)
    models = {k:dict(sum_nll=0., sum_brier=0., sum_root_auc=0., sum_auc_credit=0.) for k in ("risk", "baseline")}
    for row in rows:
        actions = [a for a in ACTIONS if not geometric or row["baseline"][a] > 0]
        if not actions:
            continue
        pos = [a for a in actions if row["labels"][a] == 1]
        neg = [a for a in actions if row["labels"][a] == 0]
        n_pairs = len(pos) * len(neg)
        totals["covered_worker_turns"] += 1
        totals["actions"] += len(actions)
        totals["positive_labels"] += len(pos)
        totals["rank_roots"] += int(n_pairs > 0)
        totals["positive_negative_pairs"] += n_pairs
        credits = {}
        for name, accum in models.items():
            p = row[name]
            for a in actions:
                y = row["labels"][a]
                accum["sum_brier"] += (p[a] - y)**2
                accum["sum_nll"] -= math.log(max(EPS, p[a] if y else 1. - p[a]))
            if n_pairs:
                credit = sum(1. if p[i] > p[j] else .5 if p[i] == p[j] else 0. for i in pos for j in neg)
                accum["sum_auc_credit"] += credit
                accum["sum_root_auc"] += credit / n_pairs
                credits[name] = credit
        if n_pairs:
            delta = credits["risk"] - credits["baseline"]
            group = "improved" if delta > 0 else "worsened" if delta < 0 else "unchanged"
            totals["root_rank_"+group] += 1
    return _finish_predict(totals, models)


def _decisions(rows):
    out = dict(eligible_worker_turns=len(rows), risk_threshold_met=0, changed=0, no_qualifying_alternative=0,
               lethal_to_safe=0, safe_to_lethal=0, lethal_to_lethal=0, safe_to_safe=0,
               original_enemy_safe_but_changed=0, originally_alive_changed=0, originally_dead_changed=0,
               actual_friendly_destination_overlap_original=0, actual_friendly_destination_overlap_alternative=0,
               original_enemy_lethal_actual_alive=0, original_enemy_safe_actual_dead=0,
               changed_cargo_sum=0., carried_on_lethal_to_safe=0., carried_on_safe_to_lethal=0.,
               captured_if_original=0., captured_if_alternative=0., changed_farther_from_home=0,
               changed_nearer_home=0, changed_same_home_distance=0, distance_increase_sum=0,
               distance_decrease_sum=0, exactly_one_extra_home_step=0)
    for row in rows:
        a = row["action"]
        probs = row["risk"]
        y = row["labels"][a]
        out["original_enemy_lethal_actual_alive"] += int(bool(y) and row["alive_real"])
        out["original_enemy_safe_actual_dead"] += int(not y and not row["alive_real"])
        if probs[a] < .25:
            continue
        out["risk_threshold_met"] += 1
        alternatives = [k for k in ACTIONS if k != a and row["friend"][k] == 0 and probs[a] - probs[k] >= .10 - EPS]
        if not alternatives:
            out["no_qualifying_alternative"] += 1
            continue
        # All selection fields precede the enemy action; occupancy and outcomes are audit-only.
        k = min(alternatives, key=lambda idx:(probs[idx], row["home"][idx], idx))
        new_y = row["labels"][k]
        transition = ("lethal" if y else "safe") + "_to_" + ("lethal" if new_y else "safe")
        out["changed"] += 1
        out[transition] += 1
        out["original_enemy_safe_but_changed"] += int(not y)
        out["originally_alive_changed" if row["alive_real"] else "originally_dead_changed"] += 1
        out["actual_friendly_destination_overlap_original"] += int(row["occupancy"][a])
        out["actual_friendly_destination_overlap_alternative"] += int(row["occupancy"][k])
        out["changed_cargo_sum"] += row["load"]
        if transition in ("lethal_to_safe", "safe_to_lethal"):
            out["carried_on_"+transition] += row["load"]
        out["captured_if_original"] += row["loss"][a]
        out["captured_if_alternative"] += row["loss"][k]
        d = row["home"][k] - row["home"][a]
        out["changed_farther_from_home" if d>0 else "changed_nearer_home" if d<0 else "changed_same_home_distance"] += 1
        out["distance_increase_sum"] += max(0,d)
        out["distance_decrease_sum"] += max(0,-d)
        out["exactly_one_extra_home_step"] += int(d == 1)
    out["avoided_minus_new_lethal"] = out["lethal_to_safe"] - out["safe_to_lethal"]
    out["captured_cargo_avoided_net"] = out["captured_if_original"] - out["captured_if_alternative"]
    return out


def _groups(rows):
    groups = {}
    for name, selected in (("all",rows), ("positive_cargo",[r for r in rows if r["load"]>0]),
                           ("cargo_ge_200",[r for r in rows if r["load"]>=200])):
        groups[name] = {"all_actions":_predict(selected,False), "geometric_actions":_predict(selected,True),
                        "decision":_decisions(selected)}
    return groups


def summarize(rows):
    by_match = defaultdict(list)
    identities = set()
    for item in rows:
        row = _normalize(item)
        ident = (row["match"],row["step"],row["seat"],row["uid"])
        if ident in identities:
            raise ValueError("Duplicate worker decision")
        identities.add(ident)
        by_match[row["match"]].append(row)
    grouped = {}
    for mid, parts in sorted(by_match.items()):
        markers = {(r["opponent"],r["seat"]) for r in parts}
        if len(markers) != 1:
            raise ValueError("Mixed opponents or candidate seats under one match identity")
        opponent,seat = markers.pop()
        grouped[mid] = dict(opponent=opponent, seat=seat, groups=_groups(parts))
    totals = _groups([r for parts in by_match.values() for r in parts])
    return dict(schema="one-step-enemy-collision-metrics/1",by_match=grouped,pooled=totals,
                primary="cargo_ge_200.geometric_actions.risk_minus_baseline.mean_root_auc",
                notes="Ranks compare positive/negative candidate actions within the same worker and turn; exact scores tie at 0.5. NLL clips the event likelihood at 1e-12. Change in risk minus baseline, lower NLL/Brier, higher AUC. Geometry selects baseline>0. Empty denominators are null. Decision diagnostics keep the observed other unit moves fixed and describe immediate collision and cargo effects; the distance difference is not a realized delay or final return.")
