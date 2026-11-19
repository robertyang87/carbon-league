"""Audit eight sealed native action tapes without running a policy or starting a new game."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
DEV=HERE.parent
SURV=DEV.parent
BASE=SURV.parents[2]
sys.path.insert(0,str(SURV))
import extract
sys.path.insert(0,str(BASE/"all_top5"))
from defense import audit as trace_cargo
from economy_extract import KEYS, POS
from turning_point_ledger import CashEvents, FastCarbon


def read(path):
    blob=path.read_bytes()
    return json.loads(gzip.decompress(blob) if path.suffix==".gz" else blob)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xy(p):
    return [p%15,p//15]


def snapshot(frames,actions,t):
    groups=[]
    for s,p in enumerate(frames[t][0]["observation"]["players"]):
        workers=[dict(uid=uid,xy=xy(w[0]),cargo=w[1],role=w[2],
                      move=actions[t][s].get(uid,"NONE")) for uid,w in p[2].items()]
        groups.append(dict(cash=p[0],trees=len(p[3]),workers=workers,
                           bases={b:dict(xy=xy(pos),command=actions[t][s].get(b,"NONE")) for b,pos in p[1].items()}))
    return dict(t=t,by_seat=groups)


def sum_events(events):
    by=[Counter(),Counter()]
    for kind,s,value,*detail in events:
        by[s][KEYS[POS[kind]]]+=value
        if kind=="tree_absorption":by[s]["enemy_cargo_tax_within_tree"]+=detail[2]
    return [{k:round(z[k],8) for k in (*KEYS,"enemy_cargo_tax_within_tree")} for z in by]


def remaining_distance(a,b):
    return sum(min(abs(x-y),15-abs(x-y)) for x,y in zip(a,b))


def deposit_approach(window,seat):
    deposits=[e for e in window["delivery_and_collision_cash"] if e[0]=="own_delivery" and e[1]==1-seat]
    if not deposits:
        return None
    largest=max(deposits,key=lambda e:e[2])
    uid,base=largest[3],xy(largest[4]); positions=[]
    for frame in window["states"]:
        enemy=next((w for w in frame["by_seat"][1-seat]["workers"] if w["uid"]==uid),None)
        if not enemy:
            continue
        ours=[w for w in frame["by_seat"][seat]["workers"] if w["role"]=="COLLECTOR" and w["cargo"]<enemy["cargo"]]
        nearest=min(ours,key=lambda w:(remaining_distance(w["xy"],enemy["xy"]),w["uid"]),default=None)
        positions.append(dict(t=frame["t"],enemy=enemy,enemy_distance_receiver=remaining_distance(enemy["xy"],base),
                              nearest_lighter_collector=nearest,
                              gap_to_enemy=remaining_distance(nearest["xy"],enemy["xy"]) if nearest else None))
    return dict(worker=uid,bank_xy=base,amount=largest[2],positions=positions)


def assess_edit(sim,before,tape,t,e):
    s=tape["candidate_seat"]; uid=e["uid"];actual=tape["actions"][t]
    if actual[s][uid]!=e["move"]:raise ValueError("actual candidate override differs")
    cell=before["players"][s][2][uid][0]; life=sim.config["treeLifespan"]
    result={}
    for move in ("NONE",e["move"]):
        joint=[dict(x) for x in actual]
        if move=="NONE":del joint[s][uid]
        branch=FastCarbon(sim.config).set_state(sim.get_state())
        after,events=extract.step(branch,before,sim.config,joint)
        dead,kind,cost=extract.label(before,(after,events),s,uid,cell,move,life)
        result[move]=dict(died=dead,acquisition=kind,paid=cost,
                          cash=[p[0] for p in after["players"]],
                          next_by_uid=after["players"][s][2].get(uid))
    a,b=result["NONE"],result[e["move"]]
    return dict(t=t,uid=uid,move=e["move"],xy=xy(cell),branches=result,
                saved=a["died"] and not b["died"],harmed=b["died"] and not a["died"],
                skipped_valid_purchase=a["paid"]>0 and b["paid"]==0)


def replay(tape,cert,arm):
    seed,s=tape["seed"],tape["candidate_seat"]; cfg=tape["game_configuration"]
    sim=FastCarbon(cfg).reset(seed=seed)
    if list(sim.carbon)!=tape["initial_full_carbon"]:raise ValueError("terrain mismatch")
    actions=tape["actions"]
    edits={r["step"]:r for r in cert.get("attribution",{}).get("decisions",[])}
    frames=[[dict(observation=dict(players=sim._players_obs(),step=0,full_carbon=list(sim.carbon)),action={}),dict(action={})]]
    sim.cash=CashEvents(sim.cash)
    tr=[]; counter=[]
    for t,joint in enumerate(actions):
        if t in edits:
            counter.append(assess_edit(sim,sim.full_observation(s),tape,t,edits[t]))
        before=list(sim.cash)
        sim.cash.events.clear();sim.step(*joint)
        ev=list(sim.cash.events); amt=sum_events(ev)
        if any(abs(sum(amt[p][k] for k in KEYS)-sim.cash[p]+before[p])>1e-5 for p in (0,1)):
            raise ValueError("cash ledger mismatch")
        tr.append(dict(t=t,after=list(sim.cash),events=ev,own_delta=sim.cash[s]-before[s],
                       enemy_delta=sim.cash[1-s]-before[1-s]))
        frames.append([dict(observation=dict(players=sim._players_obs(),step=t+1,full_carbon=list(sim.carbon)),action=joint[0]),
                       dict(action=joint[1])])
    if list(sim.cash)!=tape["last_post_cash"] or len(actions)!=cert["steps"]:
        raise ValueError("terminal state mismatch")
    if cert["official_reward_by_seat"]!=tape["official_score_by_seat"]:
        raise ValueError("recorded official score differs from replay certificate")
    by=sum_events([e for row in tr for e in row["events"]])
    if "cash" in cert:
        for side,official in enumerate(cert["cash"]["cash_entries_by_seat"]):
            expected=Counter()
            for kind,amount in official.items():
                expected[KEYS[POS[kind]]] += amount
            if any(abs(by[side][k]-expected[k])>1e-6 for k in KEYS):
                raise ValueError("independent official-game cash account disagrees")
    defense=trace_cargo(dict(steps=frames,configuration=cfg))
    for p in (0,1):
        z=defense["seat_totals"][p]
        by[p]["captured_into_collector"]=z["acquired_collector"]
        by[p]["captured_then_banked"]=z["collector_stolen_banked"]
        by[p]["captured_lost_to_tree_tax"]=z["collector_stolen_tax_paid_to_enemy_trees"]
    for e in counter:
        uid,t=e["uid"],e["t"];stop=min(t+30,len(actions))
        e["living_after_10"]=uid in frames[min(t+10,len(actions))][0]["observation"]["players"][s][2]
        first=next(((rr["t"]+1,ev) for rr in tr[t+1:stop] for ev in rr["events"] if
                    ev[0] in ("plant","seize") and ev[1]==s and ev[3]==uid),None)
        e["next_actual_purchase_within_30"]=None if first is None else dict(
            post=first[0],kind=first[1][0],price=-first[1][2],xy=xy(first[1][4]))
    deltas=[r["own_delta"]-r["enemy_delta"] for r in tr]
    tick=max(range(len(deltas)),key=lambda i:(abs(deltas[i]),-i))
    events=[v for v in tr[tick]["events"] if v[0]!="rounding"]
    hits=[e for e in events if e[0] in ("own_delivery","opponent_worker_delivery_to_this_base","base_collision_cash","tree_collision_cash")]
    focus={s:[],1-s:[]}
    players=frames[tick][0]["observation"]["players"]
    for h in sorted(hits,key=lambda v:-abs(v[2])):
        uid=h[3]
        side=next((i for i,p in enumerate(players) if uid in p[2]),None)
        if side is not None and uid not in focus[side]:focus[side].append(uid)
    for i in (0,1):
        if not focus[i]:
            cargo=sorted(players[i][2].items(),key=lambda v:(v[1][2]!="COLLECTOR",-v[1][1],v[0]))
            if cargo:focus[i]=[cargo[0][0]]
    window=dict(pre=tick,post=tick+1,delta_cash_lead=deltas[tick],
                own_delta=tr[tick]["own_delta"],enemy_delta=tr[tick]["enemy_delta"],
                components=sum_events(events),delivery_and_collision_cash=hits,focus=focus,
                states=[snapshot(frames,actions,k) for k in sorted({max(0,tick-j) for j in (10,5,3,1,0)})],
                nearby_cash=[dict(pre=r["t"],credit_and_costs=[ev for ev in r["events"] if ev[0] not in ("rounding","tree_absorption")],
                                  components=sum_events(r["events"])) for r in tr[max(0,tick-10):tick+1]])
    window["largest_rival_delivery_approach"]=deposit_approach(window,s)
    return dict(arm=arm,seed=seed,seat=s,terminal=cert["official_reward_by_seat"],
                opening_cash=[p[0] for p in frames[0][0]["observation"]["players"]],
                final_cash=tape["last_post_cash"],steps=cert["steps"],
                by_physical_seat=by,edits=counter,largest_abs_cash_lead_change=window,
                attack_lots=defense["lots"])


def action_divergence(parent,candidate,s,edits):
    changes=[next((t for t,(a,b) in enumerate(zip(parent["actions"],candidate["actions"])) if a[i]!=b[i]),None) for i in (0,1)]
    first=min((v for v in changes if v is not None),default=None)
    when=edits[0]["t"] if edits else None
    return dict(first_either=first,first_own=changes[s],first_examiner=changes[1-s],first_local_edit=when,
                opponent_already_different_before_first_edit=None if when is None else changes[1-s] is not None and changes[1-s]<when,
                same_observation_through_root=first, first_different_joint=None if first is None else
                dict(parent=parent["actions"][first],candidate=candidate["actions"][first]))


def main():
    c=read(DEV/"comparison.json")
    if len(c["pairs"])!=4 or c["games"]!=8:raise ValueError("all four sealed pairs required")
    result=[]
    for pair in c["pairs"]:
        seed,s=pair["seed"],pair["seat"];tapes={}; audited={}
        for arm in ("parent","candidate"):
            f=DEV/arm/f"raw_{seed}_{s}.json.gz";cert=pair[arm]
            if cert["transcript_sha256"]!=sha(f):raise ValueError("different signed tape")
            tapes[arm]=read(f);audited[arm]=replay(tapes[arm],cert,arm)
        if tapes["parent"]["initial_full_carbon"]!=tapes["candidate"]["initial_full_carbon"]:
            raise ValueError("different starting terrain")
        div=action_divergence(tapes["parent"],tapes["candidate"],s,audited["candidate"]["edits"])
        result.append(dict(seed=seed,seat=s,delta_margin=pair["delta_margin"],delta_U=pair["delta_U"],
                           divergence=div,**audited))
    docs=dict(comparison_sha256=sha(DEV/"comparison.json"),pairs=result)
    (HERE/"games.json.gz").write_bytes(gzip.compress(json.dumps(docs,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode(),mtime=0))
    from cash_report import write_report
    write_report(docs,HERE)


if __name__=="__main__":main()
