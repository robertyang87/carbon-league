"""Replay stored Master3s2 actions. Account for every cash write and find the largest opponent leads."""
import argparse
from collections import Counter
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
from fastsim import FastCarbon
from tree_win_gate import validate_game, verify_raw


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value, compressed=False):
    data = gzip.compress(compact(value), compresslevel=4, mtime=0) if compressed else (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return sha(data)


def cash_writes():
    """Dispatch on the pinned simulator's semantic write sites. Fail on any new cash operation."""
    source, start = inspect.getsourcelines(FastCarbon.step)
    sites = {
        'self.cash[p] = max(self.cash[p] - rec_collector_cost, 0)': "hire_collector",
        'self.cash[p] = max(self.cash[p] - rec_planter_cost, 0)': "hire_planter",
        'self.cash[tree_at[pos].player] += w.carbon; gets = tree_at[pos].player': "tree_collision_cash",
        'self.cash[self.base_at[pos][0]] += w.carbon; gets = self.base_at[pos][0]': "base_collision_cash",
        'self.cash[p] += w.carbon - smelt': "base_deposit",
        'self.cash[w.player] -= player_plant_cost[w.player]': "plant",
        'self.cash[w.player] -= cfg["seizeCost"]': "seize",
        'self.cash[t.player] += tree_carbon': "tree_absorption",
        'self.cash[0] = round(self.cash[0], 2)': "rounding",
        'self.cash[1] = round(self.cash[1], 2)': "rounding",
    }
    parsed = {start+i: sites[s.strip()] for i,s in enumerate(source) if s.strip() in sites}
    if len(parsed) != len(sites):
        raise ValueError("simulator cash-write event missing or ambiguous")
    return parsed


SITES = cash_writes()


class CashEvents(list):
    def __init__(self, values):
        super().__init__(values)
        self.events = []

    def __setitem__(self, seat, value):
        frame = sys._getframe(1)
        if frame.f_code != FastCarbon.step.__code__:
            raise RuntimeError("unexpected writer to cash account")
        kind = SITES[frame.f_lineno]
        change = value - self[seat]
        f = frame.f_locals
        ev = [kind, seat, float(change)]
        if kind == "base_deposit":
            worker = f["w"]
            ev[0] = "own_delivery" if worker.player == seat else "opponent_worker_delivery_to_this_base"
            ev.extend([worker.id, worker.pos])
        elif kind in ("plant", "seize", "tree_collision_cash", "base_collision_cash"):
            worker = f["w"]
            ev.extend([worker.id, worker.pos])
            if kind == "tree_collision_cash":
                ev.append(f["tree_at"][worker.pos].id)
        elif kind.startswith("hire_"):
            ev.extend([f["bid"], f["bpos"]])
        elif kind == "tree_absorption":
            tree = f["t"]
            sim = f["self"]
            enemy_absorption = 0.
            for cell in (tree.pos, *sim._nbr8[tree.pos]):
                wid = f["cell_worker"].get(cell)
                if wid is not None:
                    worker = sim.workers[wid]
                    if worker.is_collector and worker.player != seat:
                        enemy_absorption += worker.carbon*f["coll_rate"]
            ev.extend([tree.id, tree.pos, round(enemy_absorption,9)])
        if change or kind.startswith("hire_"):
            self.events.append(ev)
        super().__setitem__(seat, value)


def distances(size):
    def axis(a, b):
        return min(abs(a-b), size-abs(a-b))
    return [[axis(i%size, j%size) + axis(i//size, j//size) for j in range(size*size)] for i in range(size*size)]


def worker_snapshot(sim, distance):
    """All positions and loaded collectors' toroidal distance to their nearest own base."""
    return {w.id: [w.pos, round(w.carbon, 4), int(w.is_collector), w.player,
                   min(distance[w.pos][pos] for _,pos in sim.bases[w.player])]
            for w in sim.workers.values()}


def tree_changes(old, sim):
    result = []
    for k,(seat,pos) in old.items():
        current = sim.trees.get(k)
        if current is None:
            result.append(["expiry", k, pos, seat, None])
        elif current.player != seat:
            result.append(["seizure", k, pos, seat, current.player])
    result.extend(["new", k, t.pos, None, t.player] for k,t in sim.trees.items() if k not in old)
    return result


def sums_by_seat(events):
    seats = [Counter(), Counter()]
    for kind, seat, change, *_ in events:
        seats[seat][kind] += change
    return seats


def maximal_window(rows, leads, length, seat):
    if len(rows) < length:
        return None
    # First amongst ties: cash is rounded to cents on each turn.
    start = max(range(len(rows)-length+1), key=lambda i: round(leads[i+length]-leads[i], 2))
    end = start + length
    ledgers = [Counter(), Counter()]
    for row in rows[start:end]:
        for player, values in enumerate(row["cash_components"]):
            ledgers[player].update(values)
    keys = list(set(ledgers[0]) | set(ledgers[1]))
    parts = {k: round(ledgers[1-seat][k] - ledgers[seat][k], 6) for k in keys}
    diff = round(leads[end] - leads[start], 2)
    if abs(sum(parts.values()) - diff) > 0.0001:
        raise ValueError("window contributions do not sum to lead change")
    return dict(state_start=start, state_end=end, opponent_lead_start=leads[start], opponent_lead_end=leads[end],
                growth=diff, opponent_minus_candidate_components=dict(sorted(parts.items(), key=lambda p:(-abs(p[1]),p[0]))),
                components_by_seat=[{k:round(v,6) for k,v in sorted(entry.items())} for entry in ledgers])


def position_profile(state):
    out = []
    for seat in (0, 1):
        owned = [(wid, w) for wid,w in state["workers"].items() if w[3] == seat]
        loaded = [(wid,w) for wid,w in owned if w[2] and w[1] > 0]
        out.append(dict(cash=state["cash"][seat], workers=len(owned), collectors=sum(w[2] for _,w in owned),
                        trees=state["tree_count"][seat], loaded_collectors=len(loaded),
                        carried=round(sum(w[1] for _,w in loaded),3),
                        cargo_weighted_home_distance=(round(sum(w[1]*w[4] for _,w in loaded) /
                                                            sum(w[1] for _,w in loaded),2) if loaded else None),
                        largest_carriers=[[wid,w[0],w[1],w[4]] for wid,w in sorted(loaded,key=lambda e:(-e[1][1],e[0]))[:3]]))
    return out


def opening_events(rows):
    first = [{}, {}]
    for row in rows:
        for kind, seat, amount, *entity in row["cash_events"]:
            if kind not in ("rounding", "tree_absorption"):
                first[seat].setdefault(kind, [row["t"], round(amount,3), *entity])
        for wid,seat,cargo,beneficiary,survivor in row["collisions"]:
            first[seat].setdefault("first_collision_death", [row["t"],wid,round(cargo,3),beneficiary,survivor])
    return first


def annotate_window(window, states, rows, candidate):
    if window is None:
        return
    start = window["state_start"]
    before = max(0,start-10)
    window["setup_state_profiles_by_seat"] = {
        str(t):position_profile(states[t]) for t in sorted({before,start,window["state_end"]})}
    window["preceding_10_or_fewer_steps"] = dict(
        state_start=before, state_end=start,
        opponent_lead_growth=round((states[start]["cash"][1-candidate]-states[start]["cash"][candidate]) -
                                   (states[before]["cash"][1-candidate]-states[before]["cash"][candidate]),2),
        tree_events=[(row["t"],*event) for row in rows[before:start] for event in row["tree_changes"]],
        cargo_collision_losses=[(row["t"],wid,seat,round(cargo,3),beneficiary,survivor)
                               for row in rows[before:start]
                               for wid,seat,cargo,beneficiary,survivor in row["collisions"] if cargo>0],
    )


def lead_periods(leads):
    runs, start = [], None
    for t, v in enumerate([*leads, 0]):
        if v > 0 and start is None:
            start = t
        elif v <= 0 and start is not None:
            runs.append((start,t-1))
            start = None
    first_nonzero = next((t for t,v in enumerate(leads) if v != 0), None)
    first = runs[0][0] if runs else None
    twenty = next((a for a,b in runs if b-a+1>=20),None)
    crossed = next((a for a,b in runs if b-a+1>=20 and any(v<0 for v in leads[:a])),None)
    thresholds = {}
    for target in (500,1000):
        first_cross = next((t for t,v in enumerate(leads) if v>=target),None)
        first_10 = next((t for t in range(len(leads)-9) if min(leads[t:t+10])>=target),None)
        thresholds[str(target)] = dict(first_reached=first_cross,first_10_consecutive=first_10)
    return dict(opponent_cash_lead_thresholds=thresholds,first_non_tied_turn=first_nonzero, first_to_lead=(None if first_nonzero is None else
                "opponent" if leads[first_nonzero]>0 else "candidate"), first_opponent_lead=first,
                first_20_consecutive_opponent_leads=twenty, first_20_after_candidate_ever_led=crossed,
                final_unbroken_opponent_lead_from=(runs[-1][0] if runs and runs[-1][1]==len(leads)-1 else None),
                opponent_leading_turns=sum(v>0 for v in leads), candidate_leading_turns=sum(v<0 for v in leads),
                tied_turns_after_actions=sum(v==0 for v in leads[1:]))


def late_settlement(rows, seat):
    if len(rows) != 299:
        return dict(applicable=False, steps=len(rows), reason="terminated before t299; no full final t270–299 block")
    active, purchases = {}, []
    totals = [Counter(),Counter()]
    captures = [Counter(),Counter()]
    for row in rows[269:299]:
        t = row["t"]
        for kind,tid,_,old,_ in row["tree_changes"]:
            if kind == "expiry" and (tid,old) in active:
                active.pop((tid,old)).update(end_state=t,end_reason="expiry")
        after_tree = {cell:tid for kind,tid,cell,_,_ in row["tree_changes"] if kind in ("new","seizure")}
        collision_cash = {}
        for kind, owner, val, *detail in row["cash_events"]:
            totals[owner][kind] += val
            if kind in ("plant","seize"):
                wid, pos = detail
                tid = after_tree[pos]
                if (tid,1-owner) in active:
                    active.pop((tid,1-owner)).update(end_state=t,end_reason="seized")
                investment = dict(tree=tid,owner=owner,worker=wid,position=pos,buy_state=t,kind=kind,
                                  cost=-val,terrain_absorption=0.,enemy_cargo_absorption=0.,
                                  collision_cash=0.,end_state=299,end_reason="horizon")
                active[tid,owner] = investment
                purchases.append(investment)
            elif kind == "tree_absorption":
                tid, _, enemy_cargo = detail
                totals[owner]["absorption_from_enemy_carried"] += enemy_cargo
                totals[owner]["absorption_from_terrain"] += val-enemy_cargo
                if (tid,owner) in active:
                    investment = active[tid,owner]
                    investment["terrain_absorption"] += val-enemy_cargo
                    investment["enemy_cargo_absorption"] += enemy_cargo
            elif kind in ("tree_collision_cash","base_collision_cash"):
                if val>0:
                    collision_cash[detail[0]]=kind
                if kind=="tree_collision_cash" and (detail[2],owner) in active:
                    active[detail[2],owner]["collision_cash"]+=val
        for wid, dead, cargo, beneficiary, collector in row["collisions"]:
            if cargo <= 0:
                continue
            if beneficiary is None:
                dest="board_before_cell_cap"
            else:
                relation="friendly" if dead==beneficiary else "enemy"
                destination="surviving_collector_cargo" if collector is not None else collision_cash[wid]
                dest=f"{relation}_{destination}"
            captures[dead][dest]+=cargo
    by_seat = []
    for p in (0,1):
        items = [item for item in purchases if item["owner"]==p]
        for item in items:
            item["gross_received"] = sum(item[k] for k in ("terrain_absorption","enemy_cargo_absorption","collision_cash"))
            item["net_received_less_purchase"] = item["gross_received"]-item["cost"]
            item["repaid"] = item["net_received_less_purchase"]>=-1e-7
            for k,v in list(item.items()):
                if type(v) == float:
                    item[k] = round(v,6)
        buy_summary={}
        for kind in ("plant","seize"):
            selected=[x for x in items if x["kind"]==kind]
            buy_summary[kind] = dict(count=len(selected),repaid=sum(x["repaid"] for x in selected),
                                     cost=round(sum(x["cost"] for x in selected),4),
                                     received=round(sum(x["gross_received"] for x in selected),4))
        by_seat.append(dict(cash_components={k:round(v,6) for k,v in sorted(totals[p].items())},
                            outgoing_cargo_on_collisions_by_destination={k:round(v,6) for k,v in sorted(captures[p].items())},
                            tree_investments=buy_summary))
    return dict(applicable=True, state_start=269,state_end=299,by_seat=by_seat,purchases=purchases)


def audit_game(folder, game, pack, out):
    verify_raw(folder, game, pack)
    validate_game(game)
    raw = json.loads(gzip.decompress((folder/game["raw_file"]).read_bytes()))
    seed, seat = raw["seed"], raw["candidate_seat"]
    actions = raw["actions"]
    if sha(compact(actions)) != game["actions_sha256"]:
        raise ValueError("action digest mismatch")
    sim = FastCarbon(raw["game_configuration"]).reset(seed=seed)
    sim.cash = CashEvents(sim.cash)
    distance = distances(sim.size)
    init = dict(cash=list(sim.cash), workers=worker_snapshot(sim,distance), bases=sim.bases,
                carbon=sim.carbon.copy(), tree_count=[len(sim.player_trees[s]) for s in (0,1)], trees={k:[t.player,t.pos,t.age] for k,t in sim.trees.items()})
    leads = [round(sim.cash[1-seat]-sim.cash[seat],2)]
    steps = []
    max_residual = 0.
    cumulative = [Counter(),Counter()]
    checkpoint_comparisons = 0
    for t, commands in enumerate(actions, 1):
        cash0 = list(sim.cash)
        trees0 = {k:(v.player,v.pos) for k,v in sim.trees.items()}
        sim.cash.events = []
        sim.step(*commands)
        events = sim.cash.events
        attributed = sums_by_seat(events)
        snapshot = game["final"] if t == len(actions) else game.get("snapshots",{}).get(str(t))
        for p in (0,1):
            cumulative[p].update(attributed[p])
            absorption = sum(sim.trees_dict[tid][1] for tid in sim.player_trees[p])
            if abs(absorption-attributed[p]["tree_absorption"])>1e-7:
                raise ValueError("tree observer differs from engine's per-tree published absorption")
            if snapshot is not None:
                role="candidate" if p==seat else "opponent"
                for category, expected in (("own_delivery","own_positive_delivery_carbon"),
                                           ("tree_absorption","cumulative_tree_absorption_income")):
                    if abs(cumulative[p][category]-snapshot[role][expected])>1e-6:
                        raise ValueError(f"{folder.name} {seed}/{seat} t{t}: external {expected} differs")
                    checkpoint_comparisons+=1
        residual = [(sim.cash[s] - cash0[s]) - sum(attributed[s].values()) for s in (0,1)]
        max_residual = max(max_residual, *(abs(x) for x in residual))
        if max_residual > 1e-7:
            raise ValueError(f"unexplained cash movement in {folder.name}, {seed}, {seat}, {t}: {residual}")
        lead = round(sim.cash[1-seat]-sim.cash[seat], 2)
        leads.append(lead)
        steps.append(dict(t=t, actions=commands, cash=list(sim.cash), opponent_lead=lead,
                          cash_components=[dict((k,round(v,9)) for k,v in seat_values.items()) for seat_values in attributed],
                          cash_unexplained=[round(x,10) for x in residual],
                          cash_events=[[k,s,round(float(v),9),*rest] for k,s,v,*rest in events],
                          collisions=[list(l) for l in sim.last_collisions], workers=worker_snapshot(sim,distance),
                          tree_changes=tree_changes(trees0,sim), tree_count=[len(sim.player_trees[s]) for s in (0,1)]))
    state_hash = sha(compact(sim.get_state()))
    if state_hash != game["final_state_sha256"] or list(sim.cash) != raw["last_post_cash"]:
        raise ValueError(f"replayed final state differs {folder}/{seed}/{seat}")
    states = [init, *steps]
    windows = {n:maximal_window(steps,leads,n,seat) for n in (1,10,20)}
    for window in windows.values():
        annotate_window(window,states,steps,seat)
    fname=f"{folder.name}_{seed}_{seat}.json.gz"
    summary = dict(run=folder.name, seed=seed, seat=seat, turns=len(steps), result=raw["result"],
                   last_cash_by_seat=list(sim.cash), official_score_by_seat=raw["official_score_by_seat"],
                   extinct_by_seat=raw["extinct_by_seat"], official_opponent_lead=round(-raw["margin"],2),
                   terminal_scoring_adjustment_to_opponent_lead=round(-raw["margin"]-leads[-1],2),
                   last_30=late_settlement(steps,seat),
                   max_abs_unexplained_cash_residual=max_residual, external_snapshot_comparisons=checkpoint_comparisons,
                   lead_periods=lead_periods(leads),
                   windows=windows, opening=dict(
                       first_observed_economic_events_by_seat=opening_events(steps),
                       positions_by_seat={str(t):position_profile(states[t]) for t in (0,5,10,20,40) if t<len(states)}),
                   transcript_sha256=game["raw_sha256"], terminal_state_sha256=state_hash, detail=fname)
    summary["detail_sha256"]=write(out/fname,dict(run=folder.name,seed=seed,seat=seat,initial=init,steps=steps,
                                               transcript_sha256=game["raw_sha256"],terminal_state_sha256=state_hash),True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",nargs="+",type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if len(set(p.name for p in args.runs))!=len(args.runs):
        parser.error("duplicate run name")
    args.out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for run in sorted(args.runs):
        r=json.loads((run/"results.json").read_bytes())
        if r["status"]!="complete":
            raise ValueError(f"incomplete run {run}")
        seen=set()
        for g in r["games"]:
            key=(g['seed'],g['candidate_seat'])
            if key in seen:
                raise ValueError(f"duplicate {run}/{key}")
            seen.add(key)
            summaries.append(audit_game(run,g,r["header"]["pack_sha256"],args.out))
        print(f"{run.name}: {len(seen)} scored action histories verified",flush=True)
    write(args.out/"summary.json",dict(
        schema="sapling-turning-points/1", simulator_sha256=sha((ROOT/"code/fastsim.py").read_bytes()),
        analyzer_sha256=sha(Path(__file__).read_bytes()), games=summaries,
        definitions=dict(
            time="Initial state t0; step t settles the two actions chosen at state t-1. Windows exclude the start state and include the end state. Earliest among equal maxima; windows overlap across lengths.",
            lead="Opponent cash minus candidate cash, seats in unmodified seat0/seat1 order. First sustained = 20 consecutive strictly positive after-turn states. Ties interrupt streaks. Every game is included.",
            terminal="Extinction adjustment is added only to official score, excluded from operating cash windows. A negative net winner-transfer or cancelled purchase must not be inferred from gross cargo.",
            cash_events="[kind, credited/debited seat, signed actual cash change, entity ID and cell if applicable]. Trees include terrain and enemy-cargo absorption. Deposits are net of smelt and may include cargo from a prior collision; collision cash excludes cargo intercepted by a surviving collector. Rounding is separately charged.",
            workers="workers[id]=[position, cargo, is_collector, seat, nearest-own-base toroidal distance]. Impassable threats and actual future moves are excluded from geometric distance. Collision row=[dead ID, owner, pre-settlement cargo, beneficiary or null, surviving-collector seat or null].",
            scope="Complete transcript and score per listed run. Games from a shared map and two seats are dependent. These trajectories are observational; action alternatives have no labels here.",
            last_30="States 269 to 299 inclusive of 30 subsequent transitions. Every late plant or seizure is followed in its first continuous tenure until expiry, opponent seizure or end, credited only actual cash from terrain, enemy cargo absorption and tree collision. Direct cash categories carry signs; absorption_from_* subdivide tree_absorption, so do not add them twice. The opening t0 tree inventory is already included in the all-owner cash totals. Intercepted cargo on surviving workers is not cash. Treasury rounding is accounted separately at player level; no sale credit."
        )))
    print(f"total {len(summaries)}",flush=True)


if __name__ == "__main__":
    main()
