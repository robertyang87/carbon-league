"""Rebuild the new it720 vs Master6 game and independently reconcile all treasury writes."""
from collections import Counter
from pathlib import Path
import argparse
import gzip
import hashlib
import json
from latest5_freight import game_stats,ROUND,money
from turning_point_ledger import CashEvents,FastCarbon

KEY="5bd3791f228a5dda51411c62a10b6bdf-1"
CATEGORIES={"own_delivery":"own_delivery","opponent_worker_delivery_to_this_base":"other_cash",
           "tree_absorption":"absorption","tree_collision_cash":"collision_cash","base_collision_cash":"collision_cash",
           "plant":"tree_spend","seize":"tree_spend","hire_collector":"recruit","hire_planter":"recruit","rounding":"rounding"}


def cash_ledger(env):
    frames=env["steps"]
    cfg=env["configuration"]
    sim=FastCarbon(cfg).load_observation(frames[0][0]["observation"])
    sim.cash=CashEvents(sim.cash)
    records=[];positions=[];first=[{},{}];intercepts=[]
    for j,f in enumerate(frames):
        board=f[0]["observation"]["players"]
        if j:
            sim.cash.events.clear()
            sim.step(*[s.get("action") or {} for s in f])
            got=sim.full_observation()["players"]
            if any(abs(got[s][0]-board[s][0])>.011 or got[s][2].keys()!=board[s][2].keys() or
                   got[s][3]!=board[s][3] or any(x[0]!=board[s][2][u][0] or abs(x[1]-board[s][2][u][1])>.015 for u,x in got[s][2].items()) for s in (0,1)):
                raise ValueError(f"Replay diverged at {j}")
            acc=[Counter(),Counter()]
            for kind,s,val,*detail in sim.cash.events:
                acc[s][CATEGORIES[kind]]+=val
                if kind=="tree_absorption":
                    acc[s]["of_absorption_enemy_cargo"]+=detail[2]
                if val and kind not in ("rounding","tree_absorption"):
                    first[s].setdefault(kind,dict(t=j,cash=ROUND(val),detail=detail))
            for uid,owner,cargo,beneficiary,winner in sim.last_collisions:
                if cargo>0 and beneficiary==1-owner:
                    acc[1-owner]["opponent_cargo_seized_gross"]+=cargo
                    prior=frames[j-1][0]["observation"]["players"][owner][2][uid]
                    order=(f[owner].get("action") or {}).get(uid) or "NONE"
                    intercepts.append(dict(t=j,victim=uid,victim_side=owner,prior_cell=prior[0],
                        command=order,cargo=ROUND(cargo),recipient=beneficiary,
                        same_step_base_receipts=[event for event in sim.cash.events if "delivery" in event[0] and event[2]>0]))
            records.append(acc)
        positions.append(dict(t=j,cash=[v[0] for v in board],workers=[len(v[2]) for v in board],
                     collectors=[sum(w[2]=="COLLECTOR" for w in v[2].values()) for v in board],
                     trees=[len(v[3]) for v in board],cargo=[ROUND(sum(w[1] for w in v[2].values())) for v in board]))
    def phase(a,b):
        ledger=[Counter(),Counter()]
        for row in records[a:b]:
            for s in (0,1):ledger[s].update(row[s])
        for s in (0,1):
            rec=sum(ledger[s][k] for k in set(CATEGORIES.values()))
            assert abs(positions[a]["cash"][s]+rec-positions[b]["cash"][s])<.015
        diff={k:ROUND(ledger[1][k]-ledger[0][k]) for k in sorted(set(CATEGORIES.values())|{"of_absorption_enemy_cargo","opponent_cargo_seized_gross"})}
        return dict(start=a,end=b,diff=diff,margin_change=ROUND(sum(diff[k] for k in set(CATEGORIES.values()))),
                    seat_values=[{k:ROUND(v) for k,v in counter.items()} for counter in ledger])
    m=[p["cash"][1]-p["cash"][0] for p in positions]
    last_tie=max(t for t,v in enumerate(m) if v<=0)
    crossings={}
    for threshold in (0,250,500,1000,2000,5000):
        crossings[str(threshold)]=dict(first=next((i for i,v in enumerate(m) if v>threshold),None),
                sustained=next((i for i in range(len(m)) if min(m[i:])>threshold),None))
    milestones=sorted({0,10,20,30,40,50,60,90,120,150,180,210,240,270,len(frames)-1,
                       last_tie+1,*[v["sustained"] for v in crossings.values() if v["sustained"] is not None]})
    width=10
    peak=max(range(len(m)-width),key=lambda t:m[t+width]-m[t])
    return dict(verified_transitions=len(records),official_rewards=env["rewards"],first_cash_events=first,
                first_persistent_lead=last_tie+1,cash_thresholds=crossings,
                benchmark_states=[positions[t] for t in milestones],positive_opposing_intercepts=intercepts,
                strategic_phases=[phase(a,b) for a,b in ((0,30),(30,60),(60,100),(100,150),(150,200),(200,250),(250,len(records)))],
                opening_to_persistent_lead=phase(0,last_tie+1),peak_ten=phase(peak,peak+width),
                full=phase(0,len(records)))


def main(inventory,cache,out):
    reg=json.loads(inventory.read_bytes())
    rec=next(g for g in reg["games"] if g["id"]==KEY)
    data=(cache/rec["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest()!=rec["sha256"]:raise ValueError("zip identity")
    env=json.loads(gzip.decompress(data))
    ledger=cash_ledger(env)
    out.write_text(json.dumps(dict(id=KEY,source_sha256=rec["sha256"],seat_names=rec["seat_names"],ledger=ledger),indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(ledger["cash_thresholds"],ledger["full"],ledger["peak_ten"]["start"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("inventory","cache","out"):p.add_argument(name,type=Path)
    a=p.parse_args();main(a.inventory,a.cache,a.out)
