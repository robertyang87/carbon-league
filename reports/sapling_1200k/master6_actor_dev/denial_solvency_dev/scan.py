"""Locate last-producer exchange windows and replay legal single-unit commands against the submitted joint actions."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
BASE = ACTOR.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ACTOR / "tree_response_training/ten_map_economy"))
sys.path.insert(0, str(BASE / "all_top5"))
from census import FastCarbon, CashEvents, read, sha
from defense import DEST, MOVE
import fetch

COMMANDS = ("NONE", "UP", "RIGHT", "DOWN", "LEFT")


def summary(players, side, cfg):
    cash, bases, workers, trees = players[side]
    collectors = [uid for uid, w in workers.items() if w[2] == "COLLECTOR"]
    n_collectors = sum(w[2] == "COLLECTOR" for p in players for w in p[2].values())
    price = cfg["recCollectorCost"] + n_collectors * cfg["recCollectorIncreaseCost"]
    # This is the original elimination rule, which uses base recruitment prices.
    extinct = (not workers and not trees and (not bases or cash < min(cfg["recCollectorCost"], cfg["recPlanterCost"])))
    return {"cash":cash, "bases":len(bases),"collectors":len(collectors),"planters":len(workers)-len(collectors),
            "trees":len(trees), "collector_price":price,"can_pay_collector":cash >= price,"official_extinct":bool(extinct)}


def audit_root(sim, players, commands, cfg, target, t, counts):
    d = summary(players,target,cfg)
    counts["all_steps"] += 1
    if d["can_pay_collector"]:
        return
    counts["cash_below_collector_price"] += 1
    units = players[target][2]
    if d["collectors"] == 1:
        victim = next(uid for uid,w in units.items() if w[2] == "COLLECTOR")
        role = "last_collector"
    elif not d["collectors"] and d["planters"] == 1:
        victim = next(iter(units))
        role = "last_planter_without_collector"
    else:
        return
    counts[role] += 1
    if d["trees"]:
        counts[role+"_has_trees"] += 1
    if units[victim][1] == 0:
        counts[role+"_empty_target"] += 1
    if len(units) == 1 and not d["trees"]:
        counts[role+"_sole_worker_no_tree"] += 1
    hunter_seat = 1-target
    candidates = []
    target_positions = set(DEST[units[victim][0]])
    for hunter,worker in players[hunter_seat][2].items():
        for command in COMMANDS:
            cell = DEST[worker[0]][MOVE[command]]
            if cell in target_positions:
                candidates.append((hunter,command,cell))
    if not candidates:
        return
    counts[role+"_one_move_shared_destination"] += 1
    snapshot = sim.get_state()
    probe = FastCarbon(cfg).set_state(snapshot)
    probe.step(*commands)
    observed = probe._players_obs()
    target_command = commands[target].get(victim) or "NONE"
    target_cell = DEST[units[victim][0]][MOVE[target_command]]
    actual_hunters = [h for h,cmd,cell in candidates if cell == target_cell and (commands[hunter_seat].get(h) or "NONE") == cmd]
    row = {"t":t,"target_seat":target,"role":role,"victim":victim,"pre_target":d,
           "pre_target_worker":units[victim],"actual_target_command":target_command,
           "actual_target_cell":target_cell,"observed_contact_hunters":actual_hunters,
           "actual_victim_survived":victim in observed[target][2],"actual_target_after":summary(observed,target,cfg),
           "pre_hunter":summary(players,hunter_seat,cfg),"actual_hunter_after":summary(observed,hunter_seat,cfg),
           "possible_shared_cells":sorted({cell for _,_,cell in candidates}),"tested_at_observed_destination":0,
           "alternatives_gaining_removal":[], "tested_contacts":[]}
    if victim not in observed[target][2]:
        counts[role+"_actual_death"] += 1
        if actual_hunters:
            counts[role+"_actual_death_with_own_contact"] += 1
        return row
    for hunter,command,cell in candidates:
        original = commands[hunter_seat].get(hunter) or "NONE"
        if cell != target_cell or command == original:
            continue
        altered = [dict(a) for a in commands]
        if command == "NONE":
            altered[hunter_seat].pop(hunter,None)
        else:
            altered[hunter_seat][hunter] = command
        probe.set_state(snapshot)
        probe.step(*altered)
        row["tested_at_observed_destination"] += 1
        row["tested_contacts"].append({"hunter":hunter,"hunter_pre":players[hunter_seat][2][hunter],"move":command,
                                       "original_move":original,"hunter_survived":hunter in probe.workers,
                                       "target_survived":victim in probe.workers})
        if victim in probe.workers:
            continue
        alt_players = probe._players_obs()
        row["alternatives_gaining_removal"].append({
            "hunter":hunter,"hunter_pre":players[hunter_seat][2][hunter],"root_move":command,"original":original,
            "hunter_survives":hunter in probe.workers,"target_after":summary(alt_players,target,cfg),
            "our_after":summary(alt_players,hunter_seat,cfg)})
    if row["tested_at_observed_destination"]:
        counts[role+"_any_changed_command_contacts_observed_target"] += 1
    if row["alternatives_gaining_removal"]:
        counts[role+"_added_removal"] += 1
    return row


def scan(source, cert):
    tape = read(source)
    cfg = tape["game_configuration"]
    if (sha(source) != cert["transcript_sha256"] or tape["pack_sha256"] != cert.get("pack_sha256",tape["pack_sha256"])
            or tape["seed"] != cert["seed"] or tape["candidate_seat"] != cert["seat"]
            or tape["initial_terrain_u8_sha256"] != cert["map_sha256"]):
        raise ValueError("source hash or identity mismatch")
    sim = FastCarbon(cfg).reset(seed=cert["seed"])
    sim.cash = CashEvents(sim.cash)
    counts = [Counter(),Counter()]
    windows = []
    states = []
    income = []
    snapshots = []
    for t,commands in enumerate(tape["actions"]):
        # Orbital carbon outside the actor's observation can never enter preselection.
        players = sim.obs(cert["seat"])["players"]
        states.append([summary(players,i,cfg) for i in (0,1)])
        snapshots.append(players)
        for side in (0,1):
            row = audit_root(sim,players,commands,cfg,side,t,counts[side])
            if row:
                windows.append(row)
        sim.cash.events.clear()
        sim.step(*commands)
        money = [Counter(),Counter()]
        for key,side,amount,*_ in sim.cash.events:
            money[side][key] += amount
        income.append(money)
    last = sim.obs(cert["seat"])["players"]
    states.append([summary(last,i,cfg) for i in (0,1)])
    if ([p[0] for p in last] != tape["last_post_cash"] or cert["official_reward_by_seat"] != tape["official_score_by_seat"]
            or len(tape["actions"]) != cert["steps"]):
        raise ValueError("real full-match certificate mismatch")
    for row in windows:
        t,side = row["t"],row["target_seat"]
        i = cert["seat"]
        row["direction"] = "defend" if side == i else "attack"
        row["lookback"] = [{"t":u,"public_players":snapshots[u],"commands_as_recorded":tape["actions"][u]}
                           for u in sorted({max(0,t-offset) for offset in [10,5,3,0]})]
        row["observed_followup"] = {}
        for horizon in (10,30,len(income)-t):
            end = min(len(income),t+horizon)
            strip = states[t+1:end+1]
            row["observed_followup"][str(end-t)] = {"end":end, "target_state":states[end][side],
                "own_rival_delivery":[round(sum(m[k]["own_delivery"] for m in income[t:end]),3) for k in (i,1-i)],
                "own_rival_tree":[round(sum(m[k]["tree_absorption"] for m in income[t:end]),3) for k in (i,1-i)],
                "first_target_can_pay_t":next((t+j+1 for j,x in enumerate(strip) if x[side]["can_pay_collector"]),None),
                "first_target_with_collector_t":next((t+j+1 for j,x in enumerate(strip) if x[side]["collectors"]>0),None)}
    return {"raw":str(source.relative_to(ROOT)), "raw_sha256":sha(source),"seed":cert["seed"],"map":cert["map_sha256"],
            "seat":cert["seat"],"official_rewards":cert["official_reward_by_seat"],"official_extinct":tape["extinct_by_seat"],
            "attack":counts[1-cert["seat"]],"defend":counts[cert["seat"]],"root_cases":windows}


def main():
    base = ACTOR / "tree_response_training"
    games = []
    for batch in [base, *(base / f"batches/intake_{i:02}" for i in range(2,6))]:
        rc = read(batch / "source_receipt.json")
        official = batch / "source/official_replay.json"
        if sha(official) != rc["official_sha256"]:
            raise ValueError("official source receipt changed")
        for cert in read(official)["runs"]["source"]["games"]:
            row = next(r for r in rc["games"] if [r["seed"],r["seat"]]==[cert["seed"],cert["seat"]])
            if cert["map_sha256"] != row["map_sha256"] or cert["transcript_sha256"] != row["raw_sha256"]:
                raise ValueError("certified report mismatch")
            games.append(scan(ROOT / row["raw_path"],cert))
    assert len(games) == 20 and len({g["map"] for g in games}) == 10
    if any(sorted(x["seat"] for x in games if x["map"] == g["map"]) != [0,1] for g in games):
        raise ValueError("missing seat pair")
    for g in games:
        for row in g["root_cases"]:
            for item in row["alternatives_gaining_removal"]:
                state = item["target_after"]
                if state["official_extinct"] and (state["collectors"] or state["planters"] or state["trees"]):
                    raise ValueError("invalid extinction")
    counts = {d:dict(sum((g[d] for g in games),Counter())) for d in ("attack","defend")}
    (HERE/"summary.json").write_text(json.dumps({"games":20,"maps":10,"counts":counts},indent=2)+"\n")
    payload = json.dumps(games,separators=(",",":"),allow_nan=False).encode()
    (HERE/"games.json.gz").write_bytes(gzip.compress(payload,mtime=0))


if __name__ == "__main__":
    main()
