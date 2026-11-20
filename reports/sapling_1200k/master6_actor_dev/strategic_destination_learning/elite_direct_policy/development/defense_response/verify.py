"""Replay both original roots and signed continuations in the official and cash-ledger engines."""
import os
import sys

# This directory also contains the frozen selection script named select.py.
sys.path[:] = [path for path in sys.path if os.path.realpath(path or ".") != os.path.dirname(os.path.realpath(__file__))]

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
from statistics import mean
import zipfile


HERE = Path(__file__).resolve().parent
STRATEGY = HERE.parents[2]
spec = importlib.util.spec_from_file_location("paired_final_replay", STRATEGY / "replication/closing_response/verify.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
ledger = shared.ledger
need = shared.need
sha = shared.sha
digest = shared.digest
PLAN_SHA = "1031f78cff586896fd9c94adfbc2dedc3afdf1fbfc28f1eff58fa68db8a91690"
DEFENDER = "player-0-worker-11"
TARGET = "player-1-worker-22"
INPUT_SHA = "a4f923c61aa139f70019fcd2752f718e1de4c9c09b4c22940ec3ea33cd747717"
ELITE = "5e4f5da3fd63962ed559ba9efdeb788f454e3f6fb9ce617a3b5c877080795cb2"
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
MOVES = {"NONE": (0,0), "UP": (-1,0), "RIGHT": (0,1), "DOWN": (1,0), "LEFT": (0,-1)}


def move(pos, direction):
    r,c = divmod(pos, 15)
    dr,dc = MOVES[direction or "NONE"]
    return (r+dr)%15*15+(c+dc)%15


def dist(a, b):
    y, x = divmod(a,15)
    v, u = divmod(b,15)
    return min((y-v)%15,(v-y)%15)+min((x-u)%15,(u-x)%15)


def file_sha(path):
    return sha(path.read_bytes())


def changed_uid(p):
    return p["defender_uid"]


def corridor(players, issued):
    # Enumerate from simultaneous public inputs and the learner order alone.
    own, rival = players
    position, cargo, kind = rival[2][TARGET]
    need(kind == "COLLECTOR" and cargo > 0, "unloaded or ineligible target")
    bases = sorted(rival[1].values())
    def home_distance(cell):
        return min(dist(cell, base) for base in bases)
    tiles = sorted({move(position, a) for a in MOVES if home_distance(move(position,a)) < home_distance(position)})
    destinations = {k: move(w[0], issued.get(k)) for k,w in own[2].items()}
    recruits = {cell for k,cell in own[1].items() if k in issued}
    possible = []
    for uid,(origin,weight,worker_type) in own[2].items():
        if weight >= cargo or worker_type != "COLLECTOR":
            continue
        reserved = recruits | {cell for key,cell in destinations.items() if key != uid}
        for cmd in MOVES:
            cell = move(origin,cmd)
            if cell in tiles and cell not in reserved and cell != destinations[uid]:
                rivals = sorted(k for k,w in rival[2].items() if w[2] == "COLLECTOR" and w[1] <= weight and dist(w[0],cell)<=1)
                possible.append(dict(uid=uid,pos=origin,carried=weight,action=cmd,tile=cell,rival_lighter_at_one_step=rivals))
    possible.sort(key=lambda a:(a["carried"],a["tile"],a["uid"],list(MOVES).index(a["action"])))
    return dict(target=TARGET,position=position,cargo=cargo,enemy_bases=bases,closer_destinations=tiles,candidates=possible)


def check_legal(p,players):
    orders=p["anchor_orders"]
    card=corridor(players,orders["issued"])
    need(card == p["observed_checks"][-1]["inputs"] and len(card["candidates"]) == 1, "mismatching public opportunity")
    opt=card["candidates"][0]
    actual=dict(orders["issued"])
    if opt["action"]=="NONE":
        actual.pop(opt["uid"],None)
    else:
        actual[opt["uid"]]=opt["action"]
    need(opt["uid"]==p["defender_uid"]==DEFENDER and actual==orders["alternative"], "extra intervened orders")
    return card


def source():
    plan_file = HERE/"protocol.json"
    p = json.loads(plan_file.read_bytes())
    need(file_sha(plan_file)==PLAN_SHA,"diagnostic registration changed")
    path=HERE.parent/"hosted/14592050176.zip"
    need(file_sha(path)==INPUT_SHA==p["zip_sha256"],"reference artifact changed")
    doc=json.loads((HERE.parent/"hosted/settlement.json").read_bytes())
    need(doc["records_reconciled"] and doc["workflow_run"]==44651865095 and
         any(a["sha256"]==INPUT_SHA and (a["seed"],a["seat"])==(621758521,0)
             for a in doc["archives"]),"source run or map not reconciled")
    with zipfile.ZipFile(path) as z:
        raw=z.read("elite/turns.jsonl")
        rec=json.loads(z.read("elite/receipt.json"))
        reg=json.loads(z.read("elite/registration.json"))
    need(reg["seed"]==p["seed"]==621758521 and (reg["arm"],reg["candidate_seat"])==("elite",0) and
         int(reg["workflow_run"])==44651865095 and reg["physical_map_sha256"]==p["terrain"] and
         reg["files"]["candidate"]["sha256"]==ELITE and reg["files"]["opponent"]["sha256"]==ORIGINAL and
         rec["status"]=="complete" and sha(raw)==p["turns_sha256"]==rec["turns_sha256"],"wrong archived game")
    tape=[json.loads(r) for r in raw.splitlines()]
    need(len(tape)==p["terminal_step"]==rec["official_transitions"]==299,"wrong replay length")
    spec_sel=importlib.util.spec_from_file_location("frozen_defense_selector",HERE/"select.py")
    selector=importlib.util.module_from_spec(spec_sel)
    spec_sel.loader.exec_module(selector)
    pr, other_tape, other_reg, other_receipt=selector.read_source()
    need(file_sha(HERE/"select.py")==p["selector_sha256"] and other_tape==tape and other_reg==reg and
         other_receipt==rec and pr["sha256"]==INPUT_SHA,"registry/source disagreement")
    need(p["student_sha256"]==ELITE and p["original_sha256"]==ORIGINAL and p["target_uid"]==TARGET and
         p["anchor_orders"]["issued"]==tape[p["root_step"]]["calls"][0]["response"]["action"] and
         p["anchor_orders"]["enemy"]==tape[p["root_step"]]["calls"][1]["response"]["action"],"different registered actors/orders")
    return p,tape,rec


def target_events(sim, step, side, uid, before, order, after):
    """Recipients for a killed worker are recorded by the sim during collision resolution."""
    collisions=[list(e) for e in sim.last_collisions if e[0]==uid]
    payments=[list(e) for e in sim.cash.events if e[0] in ("own_delivery","opponent_worker_delivery_to_this_base")
              and e[3]==uid and e[2]>0]
    if not before and not after:
        return None
    return dict(step=step,seat=side,uid=uid,before=before,order=order.get(uid,"NONE"),after=after,
                death=before is not None and after is None,collision=collisions,positive_bank_payments=payments)


def summarize_target(events):
    payments=[(r["step"],e) for r in events if r for e in r["positive_bank_payments"]]
    own_payments=[(t,e) for t,e in payments if e[0]=="own_delivery"]
    deaths=[r["step"] for r in events if r and r["death"]]
    need(len(deaths)<=1, "same worker died more than once")
    return dict(first_own_bank=own_payments[0] if own_payments else None, payments=payments,
                total_own_bank=sum(e[2] for _,e in own_payments),death=deaths[0] if deaths else None,
                collisions=[dict(t=r["step"],outcomes=r["collision"]) for r in events if r and r["collision"]])


def replay(p, tape, receipt, ticket=None):
    cfg=ledger.config()
    random.seed(p["seed"])
    ledger.np.random.seed(p["seed"])
    env=ledger.make("carbon",configuration={**cfg,"randomSeed":p["seed"]})
    sim=ledger.FastCarbon(cfg).reset(seed=p["seed"])
    need(sha(bytes(env.states[0].observation.full_carbon))==sha(bytes(map(int,sim.carbon)))==p["terrain"],"wrong physical map")
    sim.cash=ledger.CashEvents(sim.cash)
    bank=[5.,5.]
    accounts=[Counter(),Counter()]
    trace=[]
    physical_root=None
    root_mechanism=None
    root_cash=None
    enemy_hashes=[]
    nsteps=p["root_step"]+len(ticket["rows"]) if ticket is not None else p["terminal_step"]
    for t in range(nsteps):
        need(not env.done and env.states[0].observation.step==t,"incorrect termination or timing")
        shared.check_physical(env,sim)
        published=tape[t]
        marker=ticket["rows"][t-p["root_step"]] if ticket and t>=p["root_step"] else None
        observed=shared.observations(env,bank)
        state=shared.physical_key(env)
        players=env.states[0].observation.players
        before=list(sim.cash)
        rival_decision_ids=sorted(players[1][1].keys()|players[1][2].keys())
        if marker is None:
            need(published["step"]==t and observed==published["public_observation_sha256"] and
                 all(c["status"]=="ok" for c in published["calls"]), "wrong official public tape")
            orders=[c["response"]["action"] for c in published["calls"]]
            next_bank=published["overage_remaining_seconds"]
        else:
            need(t==marker["step"] and observed==marker["public_hashes"] and state==marker["board_before"]
                 and marker["cash_before"]==before and bank==marker["bank_before"],"wrong branch public/physical state")
            orders=marker["orders"]
            clocks=[marker["student_clock"],marker["enemy_clock"]]
            need(all(math.isfinite(c["wall"]) and c["wall"]>=0 for c in clocks),"invalid clocks")
            next_bank=[x-max(0,c["wall"]-1.) for x,c in zip(bank,clocks,strict=True)]
            need(next_bank==marker["bank_after"],"inaccurate overage measurement")
            if t==p["root_step"]:
                need(orders==[p["anchor_orders"][ticket["arm"]],p["anchor_orders"]["enemy"]],"wrong committed root orders")
                orig=published["calls"][1]
                need(clocks[1]["wall"]==orig["elapsed"] and clocks[1]["cpu"]==orig["response"]["public_cpu_seconds"] and
                     clocks[1]["K"]==orig["response"]["search_last"]["candidates"] and
                     clocks[1]["H"]==orig["response"]["search_last"]["depth"],"wrong common native root receipt")
            else:
                ec=clocks[1]
                need(ec["K"]>=1 and ec["H"]>=0 and ec["rolls"]>=0 and ec["cpu"]>=0 and
                     sum(ec.get(k,0) for k in ("searched","single","aborted"))>0,"missing full original call")
        if t == 206:
            early=corridor(players, published["calls"][0]["response"]["action"])
            need(early==p["observed_checks"][0]["inputs"] and not early["candidates"],"earlier admissible corridor")
        if t==p["root_step"]:
            physical_root=state
            root_cash=before
            need(observed==p["root_public_sha256"],"wrong public input at intervention")
            root_mechanism=check_legal(p,players)
        defender,target=changed_uid(p),TARGET
        old_own=players[0][2].get(defender)
        old_enemy=players[1][2].get(target)
        if marker:
            need(old_own==marker["defender_before"] and old_enemy==marker["target_before"], "workers differ before step")
        sim.cash.events=[]
        sim.step(*orders)
        env.step(orders)
        shared.check_physical(env,sim)
        after=env.states[0].observation.players
        flow=ledger.cash_flow(sim.cash.events)
        if marker:
            need(after[0][2].get(defender)==marker["defender_after"] and after[1][2].get(target)==marker["target_after"],
                 "workers differ after step")
            need(shared.physical_key(env)==marker["board_after"] and list(sim.cash)==marker["cash_after"],"branch post-state differs")
        for seat in (0,1):
            ledger.check_equal(sum(flow[seat].values()),sim.cash[seat]-before[seat],f"cash change t={t}, seat={seat}")
            if t>=p["root_step"]:
                accounts[seat].update(flow[seat])
        if t>=p["root_step"]:
            ev=target_events(sim,t,1,target,old_enemy,orders[1],after[1][2].get(target))
            own=target_events(sim,t,0,defender,old_own,orders[0],after[0][2].get(defender))
            trace.append(dict(t=t,target=ev,defender=own,spent_and_earned=flow,
                              state_before=state,cash_after=list(sim.cash),rival_decision_ids=rival_decision_ids))
            enemy_hashes.append(digest(orders[1]))
        bank=next_bank
    scores=[s.reward for s in env.states]
    expected=ticket["official_scores_by_seat"] if ticket else receipt["official_scores_by_seat"]
    need(env.done and scores==expected and sim.step_count==nsteps,"terminal official score mismatch")
    if ticket:
        need(scores[0]-scores[1]==ticket["margin"] and physical_root==ticket["root_board_sha256"],"bad root or margin")
    for seat in (0,1):
        accounts[seat]["terminal_scoring_adjustment"]=scores[seat]-sim.cash[seat]
        ledger.check_equal(root_cash[seat]+sum(accounts[seat].values()),scores[seat],"net official final")
    return dict(scores=scores,margin=scores[0]-scores[1],terminal_step=nsteps,physical_root=physical_root,root_cash=root_cash,
                mechanism=root_mechanism,accounts=accounts,enemy_orders_sha256=enemy_hashes,trace=trace,
                target=summarize_target([e["target"] for e in trace]),defender=summarize_target([e["defender"] for e in trace]))


def check_pair(p,tape,receipt,pair,index,historic):
    games={}
    tickets={}
    signatures=[]
    for arm in pair["order"]:
        path=HERE/"tickets"/f'{pair["pair"]}_{arm}.json.gz'
        ticket=shared.read_gz(path)
        entry,=[e for e in index["tickets"] if (e["pair"],e["arm"])==(pair["pair"],arm)]
        need(file_sha(path)==entry["sha256"] and ticket["margin"]==entry["margin"] and ticket["arm"]==arm
             and ticket["pair"]==pair["pair"] and ticket["stream_seed"]==pair["stream_seed"] and ticket["protocol"]==PLAN_SHA
             and (ticket["seed"],ticket["seat"],ticket["terrain"])==(p["seed"],p["seat"],p["terrain"])
             and 0<len(ticket["rows"])<=p["terminal_step"]-p["root_step"]
             and ticket["root_public_sha256"]==p["root_public_sha256"], "unbound source or invalid ticket")
        signatures.append((ticket["root_student_sha256"],ticket["root_private_sha256"],ticket["root_board_sha256"]))
        game=replay(p,tape,receipt,ticket)
        need(game["physical_root"]==historic["physical_root"],"different physical starting point")
        live=ticket["rows"][1:]
        c=[r["enemy_clock"] for r in live]
        game["original_live_calls"]=dict(n=len(c),single=sum(a["K"]==1 for a in c),
            H_histogram=dict(Counter(a["H"] for a in c)),K_histogram=dict(Counter(a["K"] for a in c)),
            reached_cutoff=sum(a["aborted"] for a in c),rollouts=sum(a["rolls"] for a in c),
            positive_depth=sum(a["H"]>0 for a in c),zero_depth_multi=sum(a["K"]>1 and a["H"]==0 for a in c),
            cpu_seconds=sum(a["cpu"] for a in c),wall_seconds=sum(a["wall"] for a in c),
            max_wall=max((a["wall"] for a in c),default=None),slow=sum(a["wall"]>1 for a in c))
        st=[r["student_clock"] for r in live]
        game["student_live_calls"]=dict(n=len(st),cpu_seconds=sum(a["cpu"] for a in st),wall_seconds=sum(a["wall"] for a in st),
            max_wall=max((a["wall"] for a in st),default=None),slow=sum(a["wall"]>1 for a in st))
        game["ticket_sha256"]=entry["sha256"]
        tickets[arm]=ticket
        games[arm]=game
    need(len(set(signatures))==1,"different pre-action policy/memory/board state")
    issued,alt=games["issued"],games["alternative"]
    delta=alt["margin"]-issued["margin"]
    own=ledger.minus(alt["accounts"][0],issued["accounts"][0])
    rival=ledger.minus(alt["accounts"][1],issued["accounts"][1])
    components=ledger.minus(own,rival)
    ledger.check_equal(sum(components.values()),delta,"paired official margins and ledger disagree")
    steps=[]
    for i,(r1,r2) in enumerate(zip(tickets["issued"]["rows"],tickets["alternative"]["rows"])):
        need(r1["step"]==r2["step"],"inconsistent branch times")
        if r1["orders"][1]!=r2["orders"][1]:
            diffs={uid:[r1["orders"][1].get(uid,"NONE"),r2["orders"][1].get(uid,"NONE")]
                   for uid in sorted(r1["orders"][1].keys()|r2["orders"][1].keys())
                   if r1["orders"][1].get(uid,"NONE")!=r2["orders"][1].get(uid,"NONE")}
            if diffs:
                shared_actors=set(issued["trace"][i]["rival_decision_ids"]) & set(alt["trace"][i]["rival_decision_ids"])
                steps.append(dict(t=r1["step"],physical_board_differed=r1["board_before"]!=r2["board_before"],
                                  orders_issued_alternative=diffs,changed_commands_of_shared_actors={k:v for k,v in diffs.items() if k in shared_actors}))
    need(all(e["t"]>p["root_step"] for e in steps),"untied immediate response")
    first_distinct=next((e for e in steps if e["physical_board_differed"] and e["changed_commands_of_shared_actors"]),None)
    target_change=next((e for e in steps if e["physical_board_differed"] and TARGET in e["changed_commands_of_shared_actors"]), None)
    return dict(pair=pair["pair"],sampling_stream=pair["stream_seed"],games=games,alternative_minus_issued_margin=delta,
                separate_seat_cash_components=[own,rival],margin_components=components,
                first_different_enemy_order=steps[0] if steps else None,
                first_different_choice_of_same_live_enemy_on_different_boards=first_distinct,first_target_decision_difference=target_change,
                number_different_enemy_orders=len(steps),
                live_reply_interpretation="An actual original-opponent decision on observed branch states. Its measured time-limited search can also vary. Joint action differences do not isolate intention.",
                same_root_model_memory_private_order=[*signatures[0]])


def verify():
    p,tape,rec=source()
    historic=replay(p,tape,rec)
    idx=json.loads((HERE/"results.json").read_bytes())
    need(idx["protocol_sha256"]==PLAN_SHA and idx["runner_sha256"]==file_sha(HERE/"run.py")
         and idx["prefix_sha256"]==file_sha(HERE/"prefix.json.gz") and idx["whole_games"]==0, "run registration mismatch")
    pre=shared.read_gz(HERE/"prefix.json.gz")
    need(pre["protocol"]==PLAN_SHA and pre["orig_sha"]==ORIGINAL and len(pre["rows"])==p["root_step"]+1,"bad prefix registration")
    for t,row in enumerate(pre["rows"]):
        actual=tape[t]
        need(row["step"]==t and row["public_hashes"]==actual["public_observation_sha256"] and
             row["used_archived_enemy_order"]==actual["calls"][1]["response"]["action"] and
             row["corrected"]==(row["shadow_enemy_order"]!=row["used_archived_enemy_order"]),"prefix/archived order mismatch")
    masks=pre["full_prefix_legal"]
    for arm in ("issued","alternative"):
        back={}
        slots=[m[1] for m in masks[arm]]
        need(slots==sorted(set(slots)) and all(i in range(12) for i in slots),"missing or duplicate decoder slots")
        for uid,slot,idx_act,mask in masks[arm]:
            words=list(MOVES) if slot<10 else ["NONE","RECCOLLECTOR","RECPLANTER"]
            need(0<=idx_act<len(words) and mask[idx_act],"command outside conditional decoder legality")
            if idx_act:
                back[uid]=words[idx_act]
        need(back==p["anchor_orders"][arm],"conditional labels differ from physical team action")
    need([m[:2] for m in masks["issued"]]==[m[:2] for m in masks["alternative"]],"branch slot identities diverged")
    expected=[(z["pair"],arm) for z in p["pairs"] for arm in z["order"]]
    need(len(expected)==4 and [(x["pair"],x["arm"]) for x in idx["tickets"]]==expected,"incomplete registered pairs")
    pairs=[check_pair(p,tape,rec,pair,idx,historic) for pair in p["pairs"]]
    need(pairs[0]["same_root_model_memory_private_order"][1]!=pairs[1]["same_root_model_memory_private_order"][1],
         "unmatched stream separation")
    return dict(analysis="Retrospectively chosen, already-examined map. Two independent paired sampling streams conditional on one root. No independent test or training label.",
                protocol_sha256=PLAN_SHA,verifier_sha256=file_sha(Path(__file__)),results_sha256=file_sha(HERE/"results.json"),
                prefix_sha256=file_sha(HERE/"prefix.json.gz"),prefix_observations_checked=len(pre["rows"]),
                prefix_archived_original_order_corrections=sum(r["corrected"] for r in pre["rows"]),
                full_conditional_root_order_legality=masks,source_identity_by_seat=[ELITE,ORIGINAL],local_affinity=pre["affinity"],
                timing_scope="In-process, warmed continuation calls; shared archived original root excluded from per-arm live clock totals. No platform certification.",
                collision_schema=["lost_worker_id","lost_worker_seat","cargo_lost","side_receiving_cargo_or_cash","winning_collector_seat_or_null"],
                archived_zip_sha256=INPUT_SHA,archived_turns_sha256=p["turns_sha256"],archived_whole_game=historic,
                root_map_count=1,new_whole_games=0,training_eligible=False,map_independent_inference=False,
                pair_conditional_mean_margin_change=mean(x["alternative_minus_issued_margin"] for x in pairs),pairs=pairs)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived-only",action="store_true")
    args=parser.parse_args()
    if args.archived_only:
        hp,tape,rec=source()
        hist=replay(hp,tape,rec)
        print(json.dumps({k:hist[k] for k in ("scores","margin","target","mechanism","physical_root")},allow_nan=False))
    else:
        data=verify()
        out=HERE/"verified.json.gz"
        body=json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        tmp=out.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(body,mtime=0))
        tmp.replace(out)
        print(json.dumps(dict(pairs=[x["alternative_minus_issued_margin"] for x in data["pairs"]],sha256=file_sha(out))))
