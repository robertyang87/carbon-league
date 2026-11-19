"""Re-score every locked branch and settle terminal improvement across cash components."""
from collections import Counter
import math
import statistics
import run as m


def differential(ledger, seat):
    result = Counter(ledger[seat])
    result.subtract(ledger[1-seat])
    return result


def summarize():
    R = m.R
    plan = R.load(m.PLAN)
    official = R.load(m.HERE/"results.json")
    summary = []
    for case in plan["cases"]:
        s = case["seat"]
        z = {}
        case_delta = {}
        for name in ("parent", "student"):
            p = m.HERE/f"{case['seed']}_{s}_{name}.json.gz"
            a = R.load(p)
            assert a["identity"]["binding"] == official["binding"] and a["statuses"] == ["DONE", "DONE"]
            margin = a["official_scores"][s]-a["official_scores"][1-s]
            assert math.isclose(a["margin"], margin, abs_tol=1e-8)
            case_delta[name] = {k:differential(a["cash"][k],s) for k in ("window","through_199","whole")}
            calls = a["original_calls"]
            acquisitions = [c["purchased_tree"] for c in a["cash"]["changed_orders"] if c["purchased_tree"]]
            z[name] = dict(sha256=R.sha(p), official=a["official_scores"], margin=margin,
                endpoint_cash=a["cash"]["end"], own_wall_max=max(a["active_own_public_seconds"]),
                original_response_calls=len(calls), original_wall_max=max(row["public_wall_seconds"] for row in calls),
                original_cpu=sum(row["process_cpu_seconds"] for row in calls), original_wall=sum(row["public_wall_seconds"] for row in calls),
                clock=a["clock"], branch_wall_seconds=a["branch_wall_seconds"], cpu=a["identity"]["cpu"],
                changed_turns_on_branch=a["changed_turns"],changed_unit_orders_on_branch=a["changed_units"],
                realized_window_trees_acquired_by_changed_orders=len(acquisitions),
                first_tenure_ground=sum(t["ground_rent"] for t in acquisitions),
                first_tenure_cargo_tax=sum(t["cargo_tax"] for t in acquisitions),
                immediate_tree_debits=sum(t["paid"] for t in acquisitions))
        breakdown = {}
        for segment in case_delta["student"]:
            delta = case_delta["student"][segment].copy()
            delta.subtract(case_delta["parent"][segment])
            breakdown[segment] = dict(delta)
        margin_delta = z["student"]["margin"]-z["parent"]["margin"]
        physical_delta = sum(breakdown["whole"].values())
        one = dict(seed=case["seed"],seat=s,arms=z,margin_delta=margin_delta,
            component_difference_by_horizon=breakdown,physical_vs_official_delta=margin_delta-physical_delta)
        summary.append(one)
    R.write(m.HERE/"settlement.json", dict(protocol=m.REGISTERED,results=R.sha(m.HERE/"results.json"),
       source=R.sha(m.HERE/"settlement.py"),two_historical_terrains=summary,passed_registered_gate=official["passed"],
       mean_terminal_margin_delta=statistics.mean(r["margin_delta"] for r in summary),
       interpretation="Four conditional continuations on two separate development terrains. Differential is candidate-minus-parent of own-minus-rival. Realized acquisition attribution follows each changed worker only through its acquired tree's first ownership."))
    print([(r["seed"],r["margin_delta"]) for r in summary])


if __name__=="__main__":
    summarize()
