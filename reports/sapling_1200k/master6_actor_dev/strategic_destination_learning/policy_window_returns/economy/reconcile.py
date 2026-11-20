"""Attribute both players' cash on verified TRAIN trajectories using the exact executed commands."""
import argparse
from collections import Counter, defaultdict
from statistics import mean
import gzip
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
sys.path.insert(0, str(LANE / "runner"))
import run

libpath = run.LEAGUE / "natural_original/economy.py"
spec = importlib.util.spec_from_file_location("native_cash_ledger", libpath)
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)
DEPOSITS = {"own_delivery", "opponent_worker_delivery_to_this_base"}


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def flow_totals(rows):
    totals = [Counter(), Counter()]
    for row in rows:
        for seat in (0, 1):
            totals[seat].update(row["signed_cash_by_seat"][seat])
    return totals


def capture_traces(rows):
    """Follow each captured load on its first carrier. The referee does not track individual carbon lots."""
    pending = defaultdict(list)
    lots = []
    for row in rows:
        t = row["t"]
        for wid, lost_seat, cargo, recipient, collector in row["collisions"]:
            for old in pending.pop(wid, []):
                old.update(end="carrier_lost_before_deposit", end_step=t)
        for capture in row["hostile_transfer"]:
            lot = {**capture, "step": t}
            lots.append(lot)
            if capture["destination"] == "collector":
                pending[capture["carrier"]].append(lot)
            else:
                lot.update(end="cash_on_collision", end_step=t, exact_cash=capture["amount"])
        for kind, seat, change, *who in row["cash_events"]:
            if kind not in DEPOSITS:
                continue
            for lot in pending.pop(who[0], []):
                lot.update(end=kind, end_step=t, deposit_seat=seat, entire_mixed_deposit=change)
                if not lot.get("possible_tax_steps"):
                    lot["exact_cash"] = lot["amount"]
                else:
                    lot["attributable_cash_bounds"] = [0., min(change, lot["amount"])]
        for wid in row["potentially_taxed_carriers"]:
            for lot in pending.get(wid, []):
                lot.setdefault("possible_tax_steps", []).append(t)
    for items in pending.values():
        for lot in items:
            lot.update(end="carried_at_official_terminal", end_step=rows[-1]["t"]+1)
    return lots


def replay(header, trajectory, final, name):
    cfg, seed, seat = header["configuration"], header["seed"], header["seat"]
    run.need(cfg["moveCost"] == cfg["smeltCost"] == 0, "Tracing captured cargo requires zero move/smelt cost")
    env = run.collect.begin(seed, cfg)
    sim = ledger.FastCarbon(cfg).reset(seed=seed)
    run.need(run.sha(bytes(env.states[0].observation.full_carbon)) ==
             run.sha(bytes(map(int, sim.carbon))) == header["terrain"], "Economic ledger map mismatch")
    sim.cash = ledger.CashEvents(sim.cash)
    initial = list(sim.cash)
    dists = ledger.distances(sim.size)
    margins = [initial[seat]-initial[1-seat]]
    rows = []
    for i, raw in enumerate(trajectory):
        official = env.states[0].observation
        run.need(i == raw["step"] == sim.step_count == official.step and raw["board_before"] == run.board_id(env)
                 and not env.done, "Chronology or receipts do not match pre-state")
        obs = sim.obs(0)
        run.need(obs["players"] == official.players and obs["trees"] == official.trees, "Worker or tree drift")
        actors = dict(sim.workers)  # references retain the resulting destination for a deleted mover
        trees = {tid: (v.player,v.pos) for tid, v in sim.trees.items()}
        before = list(sim.cash)
        pre = {"cash": before, "workers": ledger.worker_snapshot(sim, dists),
               "trees_by_seat": [len(sim.player_trees[s]) for s in (0,1)]}
        sim.cash.events = []
        orders = raw["orders"]
        sim.step(*orders)
        env.step(orders)
        run.need(run.board_id(env) == raw["board_after"] and list(sim.cash) == raw["cash_after"] ==
                 [p[0] for p in env.states[0].observation.players], "Post-state or cash receipt mismatch")
        events = sim.cash.events
        flow = ledger.cash_flow(events)
        for s in (0, 1):
            ledger.check_equal(sum(flow[s].values()), sim.cash[s]-before[s], "Per-turn cash conservation")
        occupants = {worker.pos: worker for worker in sim.workers.values()}
        collision_cash = {e[3]: e for e in events if e[0] in ("tree_collision_cash","base_collision_cash")}
        captures = []
        for wid, victim, cargo, receiver, collector in sim.last_collisions:
            if not cargo or receiver is None or victim == receiver:
                continue
            cell = actors[wid].pos
            record = {"victim": wid, "victim_seat": victim, "beneficiary": receiver, "cell": cell, "amount": cargo}
            if collector is None:
                e = collision_cash[wid]
                ledger.check_equal(cargo, e[2], "Collision cash")
                record.update(destination=e[0], carrier=None)
            else:
                worker = occupants[cell]
                run.need(worker.is_collector and worker.player == receiver, "Wrong collision winner")
                record.update(destination="collector", carrier=worker.id)
            captures.append(record)
        tree_changes = ledger.tree_changes(trees, sim)
        new_trees = {c[1] for c in tree_changes if c[0] == "new"}
        taxed = set()
        for tree in sim.trees.values():
            if tree.id in new_trees:
                continue
            for pos in [tree.pos, *sim._nbr8[tree.pos]]:
                carrier = occupants.get(pos)
                if carrier is not None and carrier.is_collector and carrier.player != tree.player:
                    taxed.add(carrier.id)
        rows.append({"t": i, "pre": pre, "post_cash": list(sim.cash), "commands_by_seat": orders,
                     "cash_events": events, "signed_cash_by_seat": flow, "collisions": list(sim.last_collisions),
                     "tree_changes": tree_changes, "hostile_transfer": captures,
                     "potentially_taxed_carriers": sorted(taxed)})
        margins.append(round(sim.cash[seat]-sim.cash[1-seat],2))
    run.need(run.terminal(env,seat) == final, "Original and replay official terminal differ")
    total = flow_totals(rows)
    for s in (0,1):
        total[s].update(opening_cash=initial[s], terminal_scoring_adjustment=final["scores"][s]-sim.cash[s])
        ledger.check_equal(sum(total[s].values()), final["scores"][s], "Per-game official score")
    return {"arm": name, "seed": seed, "seat": seat, "terrain": header["terrain"], "official_margin": final["margin"],
            "official_scores_by_seat": final["scores"], "terminal": final, "status": "complete", "transitions": rows,
            "cash_margin_by_state": margins, "signed_account_by_seat": total, "hostile_cargo": capture_traces(rows),
            "last30": ledger.late_settlement(rows,seat), "opponent_lead": ledger.lead_periods([-v for v in margins]),
            "terminal_workers": ledger.worker_snapshot(sim,dists), "terminal_trees": [[t.id,t.player,t.pos,t.age] for t in sim.trees.values()]}


def pivots(game, start=0):
    rows = game["transitions"]
    path = game["cash_margin_by_state"]
    pivots = {}
    for label, (lo,hi) in ledger.swing_windows(path[start:]).items():
        lo, hi = start+lo, start+hi
        index = game["seat"]
        pivots[label] = {"from":lo, "to":hi, "flow":ledger.profile(index, rows, lo, hi),
                         "lookback":{n: ledger.profile(index, rows, max(0,lo-n), lo) for n in (3,5,10)},
                         "turns": list(range(max(0,lo-10),hi))}
    return pivots


def shorten(game, root):
    turns = game["transitions"]
    records = {"game":pivots(game)}
    if root is not None:
        records["from_intervention"] = pivots(game,root)
    referenced = {i for group in records.values() for entry in group.values() for i in entry["turns"]}
    if root is not None:
        referenced.update(range(max(0,root-10),min(len(turns),root+run.WINDOW)))
    return {**{k:v for k,v in game.items() if k not in ("transitions","cash_margin_by_state")},
            "pivots": records, "state_cash_margin": game["cash_margin_by_state"],
            "relevant_steps": {i:turns[i] for i in sorted(referenced)}}


def reconcile(ticket, path, protocol):
    header = load(path/"summary.json")
    receipt = load(path/"verified.json")
    source = load(path/"source.json.gz")
    run.need((receipt["seed"],receipt["seat"],receipt["role"]) == (ticket["seed"],ticket["seat"],ticket["role"]) and
             receipt["source_sha256"] == header["source_sha256"] == run.sha((path/"source.json.gz").read_bytes()) and
             header["protocol"] == protocol and source["runner_sha256"] == header["runner_sha256"], "Unbound verified source")
    base = replay(source,source["rows"],source["terminal"],"source")
    report = {"seed":ticket["seed"], "seat":ticket["seat"], "role":ticket["role"], "protocol":protocol,
              "terrain":header["terrain"], "verified_sha256":run.sha((path/"verified.json").read_bytes()),
              "source_sha256":header["source_sha256"], "source":shorten(base,None), "roots":[]}
    report["source"]["inprocess_time_limits"] = source["inprocess_time_limits"]
    for card in header["roots"]:
        t = card["step"]
        f = path / f"root_{t}.json.gz"
        run.need(run.sha(f.read_bytes()) == card["sha256"], "Changed suffix")
        root = load(f)
        if root["status"] != "complete":
            report["roots"].append({"root_step":t,"status":root["status"],"sha256":card["sha256"]})
            continue
        arms = {a: replay(header,source["rows"][:t]+r["rows"],r["terminal"],a) for a,r in root["arms"].items()}
        contrast = ledger.comparison(arms["elite"],arms["bc"])
        baseline_drift = ledger.comparison(arms["bc"],base)
        ledger.check_equal(contrast["official_margin_delta"],root["margin_delta"],"Paired delta")
        metrics = {k:contrast["margin_components"][k] for k in ("own_delivery","ground_rent","cargo_tax","plant","seize")}
        shared = root["intervention"]
        first = next((x for x in shared["shared_prefix"] if not (x["own_orders_equal"] and x["enemy_orders_equal"])), None)
        if first is None:
            cause = "identical_executed_window"
        elif first["own_orders_equal"]:
            cause = "examiner_first_on_common_board"
        elif first["enemy_orders_equal"]:
            cause = "student_first_on_common_board"
        else:
            cause = "both_changed_on_common_board"
        branches = {a:shorten(g,t) for a,g in arms.items()}
        for a, branch in branches.items():
            branch["inprocess_time_limits"] = root["arms"][a]["inprocess_time_limits"]
        # Include the exact joint orders before relative cash swings, even when neither individual trajectory peaks there.
        for window in contrast["relative_cash_windows"].values():
            arm_window = next(iter(window["main"]["window_by_arm"].values()))
            for i in range(max(0,arm_window["first_turn"]-10),arm_window["after_last_turn_state"]):
                for a in arms:
                    branches[a]["relevant_steps"][i] = arms[a]["transitions"][i]
        report["roots"].append({"root_step":t,"status":"complete","sha256":card["sha256"],"cash_comparison":contrast,
              "bc_repeat_minus_source":baseline_drift,"margin_delta":root["margin_delta"],"win_delta":root["win_delta"],
              "intervention":shared, "first_divergence":cause, "cash_components":metrics, "arms":branches})
    complete = [r for r in report["roots"] if r["status"] == "complete"]
    both = len(complete) == len(run.ROOT_STEPS)
    mean_delta = mean(r["margin_delta"] for r in complete) if both else None
    report["matched_map"] = {
        "registered_roots": len(run.ROOT_STEPS), "observed_roots": len(complete),
        "elite_minus_bc_mean_terminal_margin": mean_delta,
        "elite_minus_bc_mean_terminal_win": mean(r["win_delta"] for r in complete) if both else None,
        "bc_repeat_minus_source_mean_terminal_margin": mean(r["bc_repeat_minus_source"]["official_margin_delta"]
            for r in complete) if both else None,
        "eligible_descriptive_verdict": ("incomplete" if mean_delta is None else "positive" if mean_delta > 0
            else "negative" if mean_delta < 0 else "tied"),
        "branches_are_whole_game_wins": False,
        "first_divergence_by_root": [r["first_divergence"] for r in complete],
        "components": {k: mean(r["cash_comparison"]["margin_components"][k] for r in complete)
                     for k in ledger.KIND} if both else None}
    return report


def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument("--receipts",type=Path,required=True)
    q.add_argument("--role",choices=("fit","hold"),required=True)
    q.add_argument("--out",type=Path,required=True)
    a=q.parse_args()
    plan=LANE/"protocol.json"
    protocol=run.sha(plan.read_bytes())
    tickets=[t for t in load(plan)["schedule"] if t["role"]==a.role]
    a.out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for t in tickets:
        name=f'{t["seed"]}_{t["seat"]}'
        path=a.receipts/a.role/name
        target=a.out/(name+".json.gz")
        if not (path/"download.json").is_file():
            summary.append({**t,"status":"awaiting_complete_archive"})
            continue
        if not (path/"verified.json").is_file():
            failed = path/"failure.json"
            summary.append({**t,"status":"execution_failure" if failed.is_file() else "missing_verified_receipt",
                            "failure": load(failed) if failed.is_file() else None,
                            "has_source":(path/"source.json.gz").is_file(),
                            "available_suffix_roots":[i for i in run.ROOT_STEPS if (path/f"root_{i}.json.gz").is_file()]})
            continue
        if target.is_file():
            record=load(target)
            run.need(record["protocol"]==protocol and record["verified_sha256"]==run.sha((path/"verified.json").read_bytes())
                     and record["analyzer_sha256"]==run.sha(Path(__file__).read_bytes()),"Prior economic ticket changed")
        else:
            record=reconcile(t,path,protocol)
            record["analyzer_sha256"]=run.sha(Path(__file__).read_bytes())
            run.save(target,record)
        fields=("root_step","status","margin_delta","win_delta","first_divergence","cash_components")
        summary.append({**t,"status":"complete","source_terminal":record["source"]["terminal"],
                        "matched_map":record["matched_map"], "file_sha256":run.sha(target.read_bytes()),"roots":[
                        {k:r[k] for k in fields if k in r} for r in record["roots"]]})
    run.save(a.out/(a.role+".json"),{"role":a.role,"protocol":protocol,"maps":summary})
    print(json.dumps(summary))


if __name__=="__main__":
    main()
