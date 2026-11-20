"""Audit pre-action interception opportunities and realized cash from training-only original trajectories."""
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import zipfile

from menu import ACTIONS, board, menu_for, select, threats

HERE = Path(__file__).resolve().parent
M6 = HERE.parents[1]
SAP = M6.parent
ROOT = SAP.parents[1]
NATIVE = M6 / "cargo_tree_action_learning_dev/native_actor_corpus"
IN = NATIVE / "league_policy/native_return_intake"
INVENTORY = IN / "win_loss_sources/inventory.json.gz"
sys.path[:0] = [str(SAP), str(ROOT / "code/env_src/carbon_challenge-master")]
from turning_point_ledger import CashEvents, FastCarbon, compact, sha, write
from zerosum_env import make


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def digest(obj):
    return sha(compact(obj))


def load(row):
    path = ROOT / row["source"]
    if row["kind"] == "raw":
        record = read(path)
        cfg, commands = record["game_configuration"], record["actions"][:row["transitions"]]
    else:
        cfg_path = NATIVE.parent / "direct_original/games/candidate/raw_621758346_1.json.gz"
        cfg = dict(read(cfg_path)["game_configuration"], agentTimeout=20, actTimeout=1, runTimeout=1500)
        with zipfile.ZipFile(path) as z:
            receipt = z.read(row["arm"] + "/receipt.json")
            assert sha(receipt) == row["official_replay_receipt"]
            tape = z.read(row["arm"] + "/turns.jsonl")
            assert sha(tape) == json.loads(receipt)["turns_sha256"]
            commands = [[c["response"]["action"] for c in json.loads(line)["calls"]] for line in tape.splitlines()]
    assert len(commands) == row["transitions"] and digest(commands) == row["actions_sha256"]
    assert cfg.get("moveCost", 0) == 0
    return cfg, commands


def geometry(obs, target):
    """Optimistic position/weight reachability over all shortest paths to tied nearest bases."""
    seat = obs["player"]
    move, dist = board(int(len(obs["carbon"])**.5))
    homes = target["nearest_bases"]
    enemy_pos, horizon = target["target_pos"], target["target_home_distance"]
    own = obs["players"][seat][2]
    lighter = {w: v for w, v in own.items() if v[2] == "COLLECTOR" and v[1] < target["target_cargo"]}
    base = set(obs["players"][0][1].values()) | set(obs["players"][1][1].values())
    reachable = set()
    unavoidable = []
    for eta in range(1, horizon):
        level = [q for q,d in enumerate(dist[enemy_pos]) if d == eta and min(dist[q][h] for h in homes) == horizon-eta]
        for q in level:
            if q in base:
                continue
            if any(dist[v[0]][q] <= eta for v in lighter.values()):
                reachable.add(q)
                if len(level) == 1:
                    unavoidable.append([eta,q])
    return dict(lighter_own_collectors=len(lighter),possible_shortest_cells=len(reachable),
                all_shortest_paths_pass_reachable_cell=unavoidable,
                warning="Timing is geometric: waiting, other workers, cargo changes, and enemy rerouting affect execution.")


def compare_future(seat, root, frames):
    t, target = root["step"], root["target"]["target_uid"]
    offer = root["offer"]
    distance = board(15)[1]
    home = root["target"]["nearest_bases"]
    first_away = None
    switches = []
    payments = []
    collisions = []
    target_harvest = []
    initial_homes = set(home)
    for frame in frames[t:t+10]:
        if collisions or payments:
            break  # A new freight trip after banking is a different economic decision.
        a = frame["workers"].get(target)
        if a:
            new_pos = frame["destinations"][target]
            if min(distance[new_pos][b] for b in home) > min(distance[a[0]][b] for b in home) and first_away is None:
                first_away = frame["step"]
            closest = {b for b in frame["enemy_bases"] if distance[new_pos][b]==min(distance[new_pos][v] for v in frame["enemy_bases"])}
            if closest.isdisjoint(initial_homes) and not switches:
                switches.append(frame["step"])
            if new_pos == a[0]:
                target_harvest.append(frame["step"])
        for kind, owner, amount, uid, *detail in frame["cash"]:
            if uid == target and kind in ("own_delivery", "opponent_worker_delivery_to_this_base"):
                payments.append([frame["step"],kind,owner,amount])
        for uid,owner,carbon,beneficiary,winner_side in frame["collisions"]:
            if uid == target:
                cell = frame["destinations"][uid]
                survivors = [w for w, dest in frame["destinations"].items() if dest==cell and w not in frame["dead"]]
                winner = survivors[0] if len(survivors)==1 and winner_side is not None else None
                direct = sum(e[2] for e in frame["cash"] if e[0] in ("tree_collision_cash","base_collision_cash") and e[3]==uid and e[1]==seat)
                realized = []
                if winner and winner_side == seat:
                    for after in frames[frame["step"]:]:
                        for e in after["cash"]:
                            if e[0]=="own_delivery" and e[1]==seat and e[3]==winner:
                                realized = [after["step"], e[2]]
                                break
                        if realized or winner in after["dead"]:
                            break
                collisions.append(dict(step=frame["step"],cell=cell,cargo=carbon,winner_side=winner_side,
                    winner_uid=winner, direct_cash_to_us=direct, later_first_own_bank_by_winner=realized or None,
                    money_scope="Later bank includes other fungible carried or collected carbon."))
    root_frame=frames[t]
    exact = None
    if offer:
        focal = offer["focal_uid"]
        where = root_frame["destinations"]
        focal_cargo = root_frame["workers"][focal][1]
        own_bases = root["public"]["players"][seat][1].values()
        m = board(15)[0]
        focal_pos = root_frame["workers"][focal][0]
        def role(order):
            action = order.get(focal,"NONE")
            destination = m[focal_pos][action]
            return dict(action=action, held_position=action=="NONE", moved_toward_own_home=min(distance[destination][v] for v in own_bases)<min(distance[focal_pos][v] for v in own_bases),
                        reached_own_base=destination in own_bases, pre_action_focal_cargo=focal_cargo)
        exact = dict(alternative_financial_choices=[role(c) for c in offer["options"]],
                     actual_team_paid_tree_orders=[e for e in root_frame["cash"] if e[1]==seat and e[0] in ("plant","seize")],
                     actual_root_focal_deposit=sum(e[2] for e in root_frame["cash"] if e[1]==seat and e[0]=="own_delivery" and e[3]==focal),
                     actual_intercept_arm=offer["intercept_arm"]=="A",enemy_actual_action=root_frame["commands"][1-seat].get(target,"NONE"),
                     target_entered_cell=where[target]==offer["cell"],focal_entered_cell=where[focal]==offer["cell"],
                     focal_died=focal in root_frame["dead"],
                     focal_lost_cargo=sum(x[2] for x in root_frame["collisions"] if x[0]==focal),
                     blocked_and_won=any(c["step"]==t and c["winner_uid"]==focal and c["winner_side"]==seat for c in collisions))
    return dict(observed_joint_turns=min(10,len(frames)-t),actual_root=exact,target_positive_own_deposit=sum(z[3] for z in payments if z[1]=="own_delivery"),
                banking_events=payments,first_distance_increasing_turn=first_away,first_nearest_base_switch=switches[:1],
                stay_turns=target_harvest,target_collisions=collisions)


def replay(row):
    config, commands = load(row)
    seed, seat = row["seed"], row["seat"]
    import numpy as np
    np.random.seed(seed)
    env = make("carbon", configuration={**config,"randomSeed":seed})
    sim = FastCarbon(config).reset(seed=seed)
    assert sha(bytes(map(int,sim.carbon))) == row["terrain"] == sha(bytes(env.states[0].observation.full_carbon))
    sim.cash = CashEvents(sim.cash)
    roots, frames, earliest = [], [], None
    move = board(sim.size)[0]
    for t, issued in enumerate(commands):
        obs = sim.obs(seat)
        public = env.states[0].observation
        assert obs["players"] == public.players and obs["carbon"] == list(env.states[seat].observation.carbon) and obs["trees"] == public.trees
        if 40 <= t <= 240:
            watch = threats(obs)
            offer = select(obs,issued[seat]) if watch else None
            if offer and earliest is None:
                earliest = dict(step=t,target=offer["target_uid"],focal=offer["focal_uid"],actions=offer["options"],
                                public_sha256=digest(obs),intercept_arm=offer["intercept_arm"])
            if t%10==0 and watch:
                # Select by current cargo / (distance+1), even when no action is feasible.
                target = watch[0]
                offer = menu_for(obs,issued[seat],target)
                roots.append(dict(step=t,target=target,observed_threats=len(watch),public=obs,
                                 geometry=geometry(obs,target),offer=offer,
                                 public_sha256=digest(obs),prefix_sha256=digest(commands[:t])))
        workers = {w.id:(w.pos,w.carbon,w.player) for w in sim.workers.values()}
        destinations = {u:move[p][issued[s].get(u,"NONE")] for u,(p,_,s) in workers.items()}
        sim.cash.events.clear()
        sim.step(*issued)
        env.step(issued)
        assert list(sim.cash) == [x[0] for x in env.states[0].observation.players]
        frames.append(dict(step=t,workers=workers,destinations=destinations,commands=issued,enemy_bases=list(obs["players"][1-seat][1].values()),
                           cash=[e for e in sim.cash.events if e[0] in ("own_delivery","opponent_worker_delivery_to_this_base","tree_collision_cash","base_collision_cash","plant","seize")],
                           collisions=list(sim.last_collisions),dead={v[0] for v in sim.last_collisions}))
    assert env.done and [s.reward for s in env.states] == row["official_scores_by_seat"]
    for r in roots:
        r["actual"] = compare_future(seat,r,frames)
    return dict(seed=seed,seat=seat,terrain=row["terrain"],source=row["source"],arm=row["arm"],
                candidate_sha256=row["candidate_sha256"],commands_sha256=row["actions_sha256"],registration=row["registration"],
                official_scores=row["official_scores_by_seat"],earliest_live_menu=earliest,roots=roots)


def summarize(rows):
    c=Counter(); cash=0.
    for r in rows:
        a=r["actual"]; c["roots"]+=1
        c["reachable_shortest_lane"]+=int(bool(r["geometry"]["possible_shortest_cells"]))
        c["all_shortest_paths_through_one_reachable_cell"]+=int(bool(r["geometry"]["all_shortest_paths_pass_reachable_cell"]))
        c["one_step_offer"]+=int(r["offer"] is not None)
        c["target_banked_10"]+=int(a["target_positive_own_deposit"]>0);cash+=a["target_positive_own_deposit"]
        c["target_moved_away"]+=int(a["first_distance_increasing_turn"] is not None)
        c["target_switched_nearest_base"]+=int(bool(a["first_nearest_base_switch"]))
        f=a["actual_root"]
        if f:
            c["intercept_issued"]+=int(f["actual_intercept_arm"])
            c["target_took_offered_cell"]+=int(f["target_entered_cell"])
            c["actual_intercept_captured_named_worker"]+=int(f["blocked_and_won"])
            c["interceptor_died_on_issued_intercept"]+=int(f["actual_intercept_arm"] and f["focal_died"])
            c["captured_target_then_capturer_banked"]+=int(f["blocked_and_won"] and any(x["step"]==r["step"] and x["later_first_own_bank_by_winner"] is not None for x in a["target_collisions"]))
    return dict(**c,total_target_banked_10=round(cash,2))


def main():
    registered = [r for r in read(INVENTORY)["rows"] if r["original_role"]=="registered_native_training"]
    assert len(registered)==52 and len({r["terrain"] for r in registered})==15
    prohibited={621758450,621758451,621758470,621758471,621758500,621758501}
    assert not {r["seed"] for r in registered} & prohibited
    games = [replay(row) for row in sorted(registered,key=lambda r:(r["seed"],r["seat"],r["source"],r["arm"] or ""))]
    unique={};one={}
    for game in games:
        for root in game["roots"]:
            key=(game["terrain"],game["seat"],root["step"],root["prefix_sha256"],root["public_sha256"])
            unique.setdefault(key,root)
            if root["offer"]:
                one.setdefault(game["terrain"],dict(seed=game["seed"],seat=game["seat"],source=game["source"],
                    arm=game["arm"],step=root["step"],public_sha256=root["public_sha256"],root=root))
    book=write(HERE/"training_cases.json.gz",dict(games=games),compressed=True)
    observed=list(unique.values()); summary=dict(trajectories=len(games),physical_maps=len({g["terrain"] for g in games}),
        slots=len(games)*21,with_threat_trajectory_slots=sum(len(g["roots"]) for g in games),unique_prefix_public_roots=summarize(observed),
        one_offer_per_physical_map=summarize([c["root"] for c in one.values()]),offered_maps=[{k:v for k,v in c.items() if k!="root"} for c in one.values()],
        source_sha256=sha(INVENTORY.read_bytes()),menu_sha256=sha((HERE/"menu.py").read_bytes()),case_sha256=book)
    write(HERE/"opportunities.json",summary);print(json.dumps({k:v for k,v in summary.items() if k!="offered_maps"}))
if __name__=="__main__":main()
