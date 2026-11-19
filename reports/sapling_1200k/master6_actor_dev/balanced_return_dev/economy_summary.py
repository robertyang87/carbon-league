"""Audit the eight sealed action tapes. No agent or opponent inference is performed."""
from collections import Counter
from pathlib import Path
from statistics import mean

import common
import worlds
from return_accounts import CashEvents

HERE = Path(__file__).resolve().parent
ARMS = ("parent", "candidate")
SIDES = ("focal", "enemy")
SPEND = ("hire_collector", "hire_planter", "plant", "seize")


def relative_summary(account):
    ans = {}
    for side in SIDES:
        row = account[side]
        writes = row["cash_writes"]
        ans[side] = {
            "official_score": row["official_score"],
            "own_carrier_delivery": row["own_carrier_banked"],
            "tree_income": row["tree_absorption"],
            "collision_cash": row["base_and_tree_collision_cash"],
            "hiring_outlay": row["paid_recruitment"],
            "tree_plant_and_seize_outlay": -sum(writes.get(k, 0.) for k in ("plant","seize")),
            "hiring_and_tree_outlay": -sum(writes.get(k, 0.) for k in SPEND),
            "terminal_adjustment": writes.get("terminal_rule_adjustment", 0.),
        }
    return ans


def replay(case):
    ticket = case["ticket"]
    world = worlds.World(ticket["seed"], "development")
    seat = ticket["seat"]
    assert world.terrain_sha256 == ticket["terrain_sha256"]
    sim = world._sim
    sim.cash = CashEvents(sim.cash)
    states, flows = [], []

    def capture():
        cash = list(sim.cash)
        margins = round(cash[seat]-cash[1-seat], 2)
        return dict(cash_by_physical_seat=cash, current_cash_margin=margins,
                    terminal_rule_margin=case["official"]["margin_by_seat"][seat] if world.terminal is not None else None)

    states.append(capture())
    writes = [Counter(),Counter()]
    for joint in case["actions"]:
        sim.cash.events.clear()
        world.advance(joint)
        step_flows = [Counter(),Counter()]
        for kind, side, value, *_ in sim.cash.events:
            step_flows[side][kind] += value
            writes[side][kind] += value
        states.append(capture())
        flows.append([dict(x) for x in step_flows])
    assert world.terminal["steps"] == case["official"]["steps"]
    for side,i in zip(SIDES,(seat,1-seat)):
        expected = case["economy"][side]["cash_writes"]
        discrepancy = {k:writes[i].get(k,0.)-expected.get(k,0.) for k in writes[i].keys()|expected.keys() if k!="terminal_rule_adjustment"}
        assert max(map(abs, discrepancy.values()),default=0.) < 1e-6
        assert abs(states[-1]["cash_by_physical_seat"][i] + expected["terminal_rule_adjustment"] - case["official"]["scores"][i]) < 1e-6
    return states, flows


def differences(left, right):
    # Omitted commands and explicit None both mean stay. Missing actors after the histories diverge
    # are intentionally not attributed to the initial policy intervention.
    return {uid:{"parent":left.get(uid),"candidate":right.get(uid)}
            for uid in sorted(left.keys() | right.keys()) if left.get(uid) != right.get(uid)}


def comparison(seed,seat,row):
    cases = {}; states = {}; flows = {}
    for arm in ARMS:
        p = HERE/"development"/f"{arm}_{seed}_{seat}.json.gz"
        assert common.sha(p)==row["arms"][arm]["tape_sha256"]
        case = common.read(p)
        assert (case["ticket"]["seed"],case["ticket"]["seat"],case["ticket"]["arm"]) == (seed,seat,arm)
        cases[arm]=case
        states[arm],flows[arm]=replay(case)
    frontier = min(len(v) for v in states.values())-1
    def obs_at(state):
        return {arm:states[arm][state] for arm in ARMS}
    def margin_difference(state):
        return round(states["candidate"][state]["current_cash_margin"]-states["parent"][state]["current_cash_margin"],2)
    first_cash = next((t for t in range(1,frontier+1) if margin_difference(t)!=0.),None)
    first_command = None
    for t in range(frontier):
        changes=[differences(cases["parent"]["actions"][t][i],cases["candidate"]["actions"][t][i]) for i in (0,1)]
        if any(changes):
            first_command=dict(decision_step=t,physical_seat_changes=changes,
                               identical_focal_pre_action_observation=cases["parent"]["shadow"][t]["public_observation_sha256"]==
                                                                      cases["candidate"]["shadow"][t]["public_observation_sha256"])
            assert first_command["identical_focal_pre_action_observation"] and not changes[1-seat]
            break
    if first_cash is not None:
        assert first_command is not None and first_command["decision_step"] < first_cash
    window=[]
    if first_cash is not None:
        for t in sorted({first_cash+d for d in (-10,-5,-3,-1,0,1,3,5,10) if 0<=first_cash+d<=frontier}):
            window.append(dict(state_step=t,relative_to_first= t-first_cash,post_action_states=obs_at(t),
                               candidate_minus_parent_cash_margin=margin_difference(t),
                               preceding_turn_writes_by_physical_seat=({a:flows[a][t-1] for a in ARMS} if t else None)))
    by_arm={a:relative_summary(cases[a]["economy"]) for a in ARMS}
    delta={side:{k:by_arm["candidate"][side][k]-by_arm["parent"][side][k]
                 for k in by_arm["parent"][side]} for side in SIDES}
    assert abs(delta["focal"]["official_score"]-delta["enemy"]["official_score"]-row["delta_margin"])<1e-6
    return dict(seed=seed,seat=seat,source_tape_sha256={a:row["arms"][a]["tape_sha256"] for a in ARMS},
                by_arm=by_arm,candidate_minus_parent=delta,official_delta_margin=row["delta_margin"],
                first_executed_command_difference=first_command,first_post_action_cash_margin_difference=first_cash,
                first_cash_difference_window=window,common_observed_horizon=frontier)


def main():
    p=HERE/"development/comparison.json"
    report=common.read(p)
    if report["status"]!="complete" or len(report["pairs"])!=4:
        raise ValueError("complete sealed two-map/two-seat development receipts are required")
    rows=[comparison(r["seed"],r["seat"],r) for r in report["pairs"]]
    summary={side:{k:mean(r["candidate_minus_parent"][side][k] for r in rows)
                   for k in rows[0]["candidate_minus_parent"][side]} for side in SIDES}
    out=dict(schema="balanced-return-development-economy-v1",official_cash_source="sealed eight full proxy games, original interpreter scores in compared receipt",
             original_master3s2_games=0,additional_games=0,compared_source_sha256=common.sha(p),
             source_sha256=common.sha(Path(__file__)),cash_scan_sha256=common.sha(common.BASE/"turning_point_ledger.py"),
             explanation="Outlays are positive paid amounts. Margin direction is focal minus enemy. State t follows command t-1. Window stops at the shorter tape. Changes after first command include endogenous responses and cannot identify a unique cause.",
             pairs=rows,mean_candidate_minus_parent=summary)
    common.write(HERE/"economy_summary.json",out)


if __name__=="__main__":
    main()
