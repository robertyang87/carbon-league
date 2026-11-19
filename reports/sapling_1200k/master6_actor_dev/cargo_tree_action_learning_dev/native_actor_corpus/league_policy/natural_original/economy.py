"""Reconcile both players' cash and the actions before margin swings in the frozen original-opponent match."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
from statistics import mean
import sys
import zipfile

HERE = Path(__file__).resolve().parent
HOSTED = HERE/"hosted"
SAP = next(p for p in HERE.parents if (p/"turning_point_ledger.py").is_file())
ROOT = SAP.parents[1]
sys.path[:0] = [str(SAP), str(ROOT/"code/env_src/carbon_challenge-master")]
import numpy as np
from zerosum_env import make
from turning_point_ledger import CashEvents, FastCarbon, compact, distances, sha, tree_changes, worker_snapshot, write, late_settlement, lead_periods

RUN = 44622271891
ARMS = ("sampled_child", "sampled_parent", "greedy_parent")
PAIRS = ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2]))
SOURCE_CFG = HERE.parents[2]/"direct_original/games/candidate/raw_621758346_1.json.gz"
CFG_HASH = "7a5967383c78324d248f4b228541d7eb30f3ac3d39bf18e74f81a011a78c4631"
KIND = ("opening_cash", "own_delivery", "opponent_worker_delivery_to_this_base",
        "ground_rent", "cargo_tax", "tree_collision_cash", "base_collision_cash",
        "hire_collector", "hire_planter", "plant", "seize", "rounding", "terminal_scoring_adjustment")


def require(x, msg):
    if not x:
        raise ValueError(msg)


def cash_flow(events):
    flow = [Counter(), Counter()]
    for kind, side, change, *who in events:
        if kind == "tree_absorption":
            tax = who[2]
            flow[side]["ground_rent"] += change-tax
            flow[side]["cargo_tax"] += tax
        else:
            require(kind in KIND, f"unclassified monetary event {kind}")
            flow[side][kind] += change
    return flow


def minus(a, b):
    return {k: a.get(k, 0.) - b.get(k, 0.) for k in KIND}


def check_equal(a, b, name):
    require(math.isclose(a, b, rel_tol=0., abs_tol=1.e-5), f"{name}: {a} vs {b}")


def profile(idx, rows, begin, end):
    """Endpoint differences and seat-specific signed flows for executed commands [begin,end)."""
    total = [Counter(),Counter()]
    for r in rows[begin:end]:
        for side in (0,1):
            total[side].update(r["signed_cash_by_seat"][side])
    return dict(first_turn=begin, after_last_turn_state=end, turns=end-begin,
                learner_margin_change=sum(total[idx].values())-sum(total[1-idx].values()),
                by_physical_seat=total)


def swing_windows(path):
    if len(path) < 2:
        return {}
    out = {}
    for width in (1,10):
        if width >= len(path):
            continue
        # Earliest tied window. Each path value is a cash difference at the
        # beginning of state t; differences are rounded to recorded cents.
        gain = [round(b-a,2) for a,b in zip(path, path[width:])]
        for name, choose in (("gain",max),("loss",min)):
            t = choose(range(len(gain)), key=gain.__getitem__)
            out[f"{name}_{width}"] = (t, t+width)
    return out


def windows(rows, seat, margins):
    result = {}
    for label,(begin,end) in swing_windows(margins).items():
        result[label] = dict(swing=profile(seat,rows,begin,end),
                             prior={n:profile(seat,rows,max(0,begin-n),begin) for n in (3,5,10)})
    return result


def config():
    source = SOURCE_CFG.read_bytes()
    require(sha(source) == CFG_HASH, "hosted environment configuration changed")
    return dict(json.loads(gzip.decompress(source))["game_configuration"], agentTimeout=20, actTimeout=1, runTimeout=1500)


def replay(g, archive, cfg):
    seed, seat, arm = g["seed"], g["seat"], g["arm"]
    rec = archive.read(arm+"/receipt.json")
    require(sha(rec)==g["receipt_sha256"], "different signed receipt")
    rec = json.loads(rec)
    member = arm+"/turns.jsonl"
    tape = archive.read(member) if member in archive.namelist() else b""
    if tape or member in archive.namelist():
        require(sha(tape)==rec["turns_sha256"], "different signed actions")
    require((g["official_transitions"],g["official_scores_by_seat"]) ==
            (rec["official_transitions"],rec.get("official_scores_by_seat")), "different verified terminal result")
    np.random.seed(seed)
    env = make("carbon", configuration={**cfg,"randomSeed":seed})
    sim = FastCarbon(cfg).reset(seed=seed)
    require(sha(bytes(map(int,sim.carbon))) == sha(bytes(env.states[0].observation.full_carbon)) == g["terrain"],
            "different physical terrain")
    sim.cash = CashEvents(sim.cash)
    initial = list(sim.cash)
    dist = distances(sim.size)
    rows = []
    accounts = [Counter(dict(opening_cash=initial[s])) for s in (0,1)]
    margin = [initial[seat]-initial[1-seat]]
    tape_rows = [json.loads(line) for line in tape.splitlines()]
    for t,row in enumerate(tape_rows[:g["official_transitions"]]):
        calls = row["calls"]
        require(t == row["step"] == sim.step_count == env.states[0].observation.step and not env.done
                and len(calls)==2 and all(c["status"]=="ok" for c in calls), "incomplete observed transition")
        commands = [c["response"]["action"] for c in calls]
        before = list(sim.cash)
        public = sim.obs(0)
        official = env.states[0].observation
        require(public["players"] == official.players and public["trees"] == official.trees,
                "worker or tree state differs from the official replay")
        trees = {key:(v.player,v.pos) for key,v in sim.trees.items()}
        workers = worker_snapshot(sim,dist)
        pre = dict(cash=before, workers=workers, trees_by_seat=[len(sim.player_trees[s]) for s in (0,1)],
                   carried_by_seat=[sum(w.carbon for w in sim.workers.values() if w.player==s and w.is_collector) for s in (0,1)])
        sim.cash.events = []
        sim.step(*commands)
        env.step(commands)
        events = sim.cash.events
        flow = cash_flow(events)
        for s in (0,1):
            check_equal(sum(flow[s].values()), sim.cash[s]-before[s], "component cash movement")
            check_equal(sim.cash[s],env.states[0].observation.players[s][0],"official post-action cash")
            accounts[s].update(flow[s])
        rows.append(dict(t=t,public_observation_sha256=row["public_observation_sha256"],
                         pre=pre,commands_by_seat=commands,post_cash=list(sim.cash),cash_events=events,
                         tree_changes=tree_changes(trees,sim), collisions=list(sim.last_collisions),signed_cash_by_seat=flow))
        margin.append(round(sim.cash[seat]-sim.cash[1-seat],2))
    require(len(rows) == g["official_transitions"], "short action tape")
    scores = g["official_scores_by_seat"]
    if scores is not None:
        require(g["status"]=="complete" and env.done and [s.reward for s in env.states]==scores, "incorrect official scores")
        for s in (0,1):
            accounts[s]["terminal_scoring_adjustment"] = scores[s]-sim.cash[s]
            check_equal(sum(accounts[s].values()), scores[s], "signed sources and official score")
    terminal_workers = worker_snapshot(sim,dist)
    terminal_assets = [dict(alive_trees=len(sim.player_trees[s]),
                            carried=sum(w[1] for w in terminal_workers.values() if w[3]==s and w[2]),
                            carriers={wid:w for wid,w in terminal_workers.items() if w[3]==s and w[2] and w[1]>0})
                       for s in (0,1)]
    phases = [profile(seat,rows,begin,min(begin+100,len(rows))) for begin in range(0,len(rows),100)]
    return dict(arm=arm,seed=seed,seat=seat,terrain=g["terrain"],status=g["status"],counted=g["counted"],
                action_tape_sha256=sha(tape),executed_commands_sha256=sha(compact([r["commands_by_seat"] for r in rows])),
                executed_transitions=len(rows),unexecuted_requests=len(tape_rows)-len(rows),winner_seat=g["winner_seat"],
                official_scores_by_seat=scores,official_margin=None if scores is None else scores[seat]-scores[1-seat],
                physical_cash_by_seat=list(sim.cash),signed_account_by_seat=accounts,cash_margin_by_state=margin,
                end_assets_by_physical_seat=terminal_assets,phases=phases,turning_points=windows(rows,seat,margin),
                opponent_lead=lead_periods([-v for v in margin]),last30_tree_and_cargo=late_settlement(rows,seat),transitions=rows)


def comparison(a,b):
    seat = a["seat"]
    require((a["seed"],a["seat"],a["terrain"]) == (b["seed"],b["seat"],b["terrain"]), "unpaired terrain or seat")
    result = dict(seed=a["seed"],seat=seat,arms=[a["arm"],b["arm"]],numeric=all(g["official_margin"] is not None for g in (a,b)),
                  statuses=[a["status"],b["status"]])
    if not result["numeric"]:
        return result
    balance = [minus(x,y) for x,y in zip(a["signed_account_by_seat"],b["signed_account_by_seat"])]
    delta = minus(balance[seat],balance[1-seat])
    official = a["official_margin"]-b["official_margin"]
    check_equal(sum(delta.values()),official,"paired signed sources and official margin")
    path = [x-y for x,y in zip(a["cash_margin_by_state"],b["cash_margin_by_state"])]
    pivots = {}
    for label,(t,u) in swing_windows(path).items():
        def paired(start,end):
            x,y = [profile(seat,g["transitions"],start,end) for g in (a,b)]
            return dict(window_by_arm={a["arm"]:x,b["arm"]:y},
                        change_in_relative_margin=round(x["learner_margin_change"]-y["learner_margin_change"],2))
        pivots[label] = dict(main=paired(t,u),before={n:paired(max(t-n,0),t) for n in (3,5,10)})
    first = {s:next((x["t"] for x,y in zip(a["transitions"],b["transitions"])
                     if x["commands_by_seat"][s]!=y["commands_by_seat"][s]),None) for s in (0,1)}
    result.update(official_margin_delta=official,learner_finances=balance[seat],opponent_finances=balance[1-seat],
                  margin_components=delta,first_different_order_by_physical_seat=first,relative_cash_windows=pivots,
                  last_comparable_state=len(path)-1,change_after_last_comparable_state=official-path[-1])
    return result


def summarize(pairs):
    def describe(group):
        valid = [p for p in group if p["numeric"]]
        if not valid:
            return dict(planned=len(group),numeric=0)
        values = {}
        for key in ("learner_finances","opponent_finances","margin_components"):
            values[key] = {k:mean(p[key][k] for p in valid) for k in KIND}
        return dict(planned=len(group),numeric=len(valid),mean_numeric_official_margin_delta=mean(p["official_margin_delta"] for p in valid),
                    mean_over_all_planned=mean(p["official_margin_delta"] for p in valid) if len(valid)==len(group) else None,**values)
    return dict(all=describe(pairs),by_learner_seat={s:describe([p for p in pairs if p["seat"]==s]) for s in (0,1)})


def main():
    settle = HOSTED/"settlement.json"
    doc = json.loads(settle.read_bytes())
    require(doc["run_id"]==RUN and doc["records_reconciled"] and len(doc["games"])==12, "waiting for verified 12-arm settlement")
    matches = {(g["seed"],g["seat"],g["arm"]):g for g in doc["games"]}
    cfg = config()
    games = {}
    for attachment in doc["archives"]:
        archive_path = HOSTED/(str(attachment["id"])+".zip")
        require(sha(archive_path.read_bytes())==attachment["sha256"], "archive hash differs")
        key = (attachment["seed"],attachment["seat"])
        with zipfile.ZipFile(archive_path) as archive:
            for arm in ARMS:
                ident = (*key,arm)
                require(ident not in games,"duplicate ticket")
                games[ident] = replay(matches[ident],archive,cfg)
    require(len(games)==12 and set(matches)==set(games),"missing archived match")
    all_pairs = sorted({(s,side) for s,side,_ in games})
    comparisons, means = {}, {}
    for new,ref in PAIRS:
        label = f"{new} minus {ref}"
        rows = [comparison(games[s,seat,new],games[s,seat,ref]) for s,seat in all_pairs]
        native = doc["descriptive_contrasts"][label]
        check_equal(sum(p["numeric"] for p in rows),native["paired_numeric_games"],"official paired count")
        means[label] = summarize(rows)
        if native["mean_over_fixed_maps"] is not None:
            check_equal(means[label]["all"]["mean_over_all_planned"],native["mean_over_fixed_maps"],"official paired mean")
        comparisons[label]=rows
    detail = HOSTED/"economy_turns.json.gz"
    rows_sha = write(detail,dict(workflow=RUN,games=[dict(seed=g["seed"],seat=g["seat"],arm=g["arm"],transitions=g.pop("transitions"))
                                                     for g in games.values()]),True)
    artifact = dict(workflow=RUN,settlement_sha256=sha(settle.read_bytes()),analyzer_sha256=sha(Path(__file__).read_bytes()),
      source_config_sha256=CFG_HASH,turns_sha256=rows_sha,turns_file=detail.name,
      interpretation="Conditional accounting of realized actions on two previously observed maps with natural sampling by the trained and reference agents. Each price and cash movement follows the official rules. Different trajectories do not isolate policy causation.",
      interval_semantics="State t is immediately before simultaneous orders at t. [start,end) means recorded source commands from start through end-1. References at -3/-5/-10 stop at the initial state. First tied max/min in cents.",
      worker_tuple="position, carried carbon, is_collector, physical owner, toroidal distance to nearest own base",
      collision_tuple="dead worker ID, physical owner of dead worker, carbon carried, beneficiary seat or null, surviving collector ID or null",
      signed_costs="Recruitment/planting/seizure are negative; ground_rent excludes the opponent cargo tax portion of tree income; collision payments are separate. Terminal adjustment equals official score less final physical cash.",
      games=list(games.values()),paired_comparisons=comparisons,means=means)
    write(HOSTED/"economy.json.gz",artifact,True)
    for k,m in means.items():
        val=m["all"]
        print(k,val.get("mean_over_all_planned"),{n:round(val.get("margin_components",{}).get(n,0),2) for n in ("own_delivery","ground_rent","cargo_tax")},flush=True)


if __name__=="__main__":
    main()
