"""Largest observed ten-transition cash swings in every frozen Top Five head-to-head."""
import argparse
from collections import Counter
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "reports/master_family/deep_review"))
from materialize_top5_h2h import ARCHIVE, IDENTITIES, read_repo
from carbon_deep_events import _actions, _reconstruct, _snapshot
from carbon_deep_space import distance


def frame_summary(frames, t):
    p = frames[t][0]["observation"]["players"]
    return dict(t=t,cash=[s[0] for s in p], workers=[len(s[2]) for s in p],
                collectors=[sum(w[2]=="COLLECTOR" for w in s[2].values()) for s in p],
                cargo=[round(sum(w[1] for w in s[2].values()),2) for s in p],
                trees=[len(s[3]) for s in p])


def watch(frames, t, seat, uid, size):
    p = frames[t][0]["observation"]["players"]
    w = p[seat][2].get(uid)
    if w is None:
        return dict(t=t, present=False)
    pos,cargo,role=w
    enemies=[(distance(pos,v[0],size),k,v) for k,v in p[1-seat][2].items()]
    # A lighter or tied enemy collector can remove this worker; for a planter any
    # enemy worker may kill it. Current-cell exposure is only a reach indicator.
    dangerous=[(d,k) for d,k,v in enemies if d<=1 and (role=="PLANTER" or
                (v[2]=="COLLECTOR" and v[1]<=cargo))]
    return dict(t=t,pos=pos,cargo=round(cargo,2),home_distance=min(distance(pos,b,size) for b in p[seat][1].values()),
                nearest_enemy=min((d for d,k,v in enemies),default=None),
                dangerous_enemy_can_reach_current_cell=sorted(k for d,k in dangerous))


def pipeline(frames,t,side,size):
    players = frames[t][0]["observation"]["players"]
    reading=[]
    for s,player in enumerate(players):
        ours=player[2]; opponents=players[1-s][2]
        loaded=[w for w in ours.values() if w[2]=="COLLECTOR" and w[1]>0]
        close=[w for w in loaded if min(distance(w[0],p,size) for p in player[1].values())<=3]
        safe=[w for w in close if not any(e[2]=="COLLECTOR" and e[1]<=w[1] and distance(e[0],w[0],size)<=1
                                          for e in opponents.values())]
        reading.append((sum(w[1] for w in close),sum(w[1] for w in safe),len(player[3])))
    names=("own_base_distance_le3_cargo","own_base_distance_le3_current_cell_unthreatened_cargo","tree_count")
    return {n:round(reading[side][j]-reading[1-side][j],2) for j,n in enumerate(names)}


def rank_window(margins):
    deltas=[margins[t+10]-margins[t] for t in range(len(margins)-10)]
    t=max(range(len(deltas)),key=lambda j:abs(deltas[j]))
    return t,0 if deltas[t]>=0 else 1


def analyze(meta, replay, pair, focus_end=None):
    f=replay["steps"]
    cfg=replay["configuration"]
    if [r[0]["observation"]["step"] for r in f] != list(range(len(f))) or replay["rewards"]!=meta["rewards"]:
        raise ValueError("step, outcome or seat mismatch "+meta["id"])
    margins=[round(100*r[0]["observation"]["players"][0][0])-round(100*r[0]["observation"]["players"][1][0]) for r in f]
    if focus_end is None:
        start,side=rank_window(margins)
    else:
        start=max(0,focus_end-10)
        side=int(meta["rewards"][1]>meta["rewards"][0])
    stop=start+10 if focus_end is None else focus_end
    sign=1-2*side
    offsets = (start-10,start-5,start-3,start,stop) if focus_end is None else (stop-10,stop-5,stop-3,stop-1,stop)
    stamps=sorted(set(max(0,t) for t in offsets))
    scores=replay["rewards"]
    result=dict(id=meta["id"],seed=meta["seed"],seat_names=meta.get("models",meta.get("seat_names")),
                end_step=len(f)-1,rewards=scores,beneficiary=side,window=[start,stop],
                margin_before=round(sign*margins[start]/100,2),margin_after=round(sign*margins[stop]/100,2),
                swing=sign*(margins[stop]-margins[start])/100,snapshots=[frame_summary(f,t) for t in stamps])
    deliveries=[[],[]]; trees=[[],[]]; collisions=[[],[]]; recruits=[[],[]]
    for t in range(start,stop):
        a,b=(_snapshot(f[i],cfg["size"]) for i in (t,t+1))
        for s in (0,1):
            for e in _reconstruct(a,b,_actions(f[t+1]),cfg,s):
                typ=e["kind"]; pos=e["position"]
                if typ in ("delivery","enemy_delivery_to_us"):
                    deliveries[s].append(dict(t=t+1,worker=e["worker_id"],carrier=e["carrier_seat"],cell=pos,
                                              amount=round(e["net_cash"],4)))
                elif typ in ("tree_planted","tree_captured","tree_expired","tree_lost_to_capture"):
                    planter=next((wid for wid,v in b["workers"].items() if v["pos"]==pos and v["type"]=="PLANTER"),None)
                    trees[s].append(dict(t=t+1,cell=pos,tree=e["tree_id"],event=typ,planter=planter))
                elif typ=="collision" and e["own_deaths"]:
                    for w in e["participants"]:
                        if w["seat"]==s and w["worker_id"] in e["deleted_worker_ids"]:
                            collisions[s].append(dict(t=t+1,cell=pos,worker=w["worker_id"],cargo=round(w["known_cargo"],2),
                                                      victor=e["winner_worker_id"],contested=e["contested"]))
                elif typ=="recruit":
                    recruits[s].append(dict(t=t+1,worker=e["worker_id"],cost=e["cost"]))
    flows=[]
    for t in range(start+1,stop+1):
        previous=f[t-1][0]["observation"]["players"]
        obs=f[t][0]["observation"]
        post=obs["players"]
        cash=[round(post[s][0]-previous[s][0],2) for s in (0,1)]
        income=[round(sum(obs["trees"][uid][1] for uid in post[s][3]),4) for s in (0,1)]
        delivery=[round(sum(d["amount"] for d in v if d["t"]==t),4) for v in deliveries]
        hire=[sum(k["cost"] for k in v if k["t"]==t) for v in recruits]
        residual=[round(cash[s]-income[s]-delivery[s]+hire[s],4) for s in (0,1)]
        flows.append(dict(t=t,cash_delta=cash,tree_absorption=income,base_deposits=delivery,recruit_cost=hire,
                          other_cash_including_tree_fees_collisions_and_rounding=residual))
    result["tree_absorption_income"]=[round(sum(r["tree_absorption"][s] for r in flows),2) for s in (0,1)]
    sums=[round(sum(d["amount"] for d in seat),2) for seat in deliveries]
    result.update(deposits_by_receiving_seat=sums,largest_deliveries_by_seat=[max(v,key=lambda e:e["amount"]) if v else None for v in deliveries],
                  delivery_net_advantage=round(sums[side]-sums[1-side],2),
                  tree_changes=[dict(Counter(e["event"] for e in v)) for v in trees],
                  collision_deaths=[len(v) for v in collisions],recruit_cost=[sum(k["cost"] for k in v) for v in recruits])
    # Always keep full identities for both seats' largest realized depositors;
    # compare their actual earlier paths before seeing the transfer.
    witnesses=[]
    for s in (0,1):
        choices=[]
        if deliveries[s]:
            v=max(deliveries[s],key=lambda e:e["amount"])
            choices.append(("delivery",v["carrier"],v["worker"],v["t"]))
        own_deliveries = [v for v in deliveries[s] if v["carrier"]==s]
        if own_deliveries:
            v=max(own_deliveries,key=lambda e:e["amount"])
            if ("delivery",s,v["worker"],v["t"]) not in choices:
                choices.append(("own_delivery",s,v["worker"],v["t"]))
        if collisions[s]:
            v=max(collisions[s],key=lambda e:e["cargo"])
            choices.append(("collision_death",s,v["worker"],v["t"]))
        if pair:
            for e in trees[s]:
                if e["planter"] and e["event"] in ("tree_planted","tree_captured"):
                    choices.append((e["event"],s,e["planter"],e["t"]))
        for event,owner,uid,when in choices:
            if any((v["event"],v["worker"],v["at"]) == (event,uid,when) for v in witnesses):
                continue
            times=sorted(set(stamps[:-1]+[max(0,when-k) for k in (10,5,3,1,0)]))
            witnesses.append(dict(event=event,worker=uid,seat=owner,at=when,path=[watch(f,t,owner,uid,cfg["size"]) for t in times]))
    result["witnesses"]=witnesses
    if focus_end is None:
        # One within-game seat-phase baseline per lag. Exclude the selected window
        # and its approach, and use no game outcome in the state's feature formula.
        probes={}
        for lag in (3,5,10):
            t=start-lag
            if t<0: continue
            control=[j for j in range(0,len(f),5) if j//100==t//100 and (j<start-10 or j>stop)]
            if not control:continue
            x=pipeline(f,t,side,cfg["size"])
            other=[pipeline(f,j,side,cfg["size"]) for j in control]
            probes[str(lag)]=dict(t=t,same_phase_control_states=len(control),before=x,
                                  control_median={k:median(y[k] for y in other) for k in x})
        result["same_game_retrospective_comparison"]=probes
    if pair:
        result["step_accounting"] = flows
        result["events"] = dict(deliveries=deliveries,tree_changes=trees,collision_deaths=collisions,recruits=recruits)
        winner=0 if scores[0]>scores[1] else 1
        if focus_end is None and margins[-1]*(1-2*winner)>0:
            last=next((j for j in range(len(margins)-2,-1,-1) if margins[j]*(1-2*winner)<=0),None)
            if last is None:
                raise ValueError("expected opening tie")
            result["eventual_winner_last_took_cash_lead"] = last+1
            result["permanent_lead_window"] = analyze(meta,replay,True,last+1)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_cache",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    idx_bytes=read_repo(ARCHIVE/"sources.json"); idx=json.loads(idx_bytes)
    a=read_repo(IDENTITIES); b=read_repo(ARCHIVE/"later7.json")
    assert sha256(a).hexdigest()==idx["frozen101_receipt_sha256"]
    assert sha256(b).hexdigest()==idx["later7_receipt_sha256"]
    roster={g["id"]:g for g in json.loads(a)["mutual_matches"]+json.loads(b)["games"]}
    assert len(roster)==108==len({g["id"] for g in idx["games"]})
    games=[]; top=[]
    for g in idx["games"]:
        data=(args.replay_cache/g["cohort"]/(g["id"]+".json.gz")).read_bytes()
        assert sha256(data).hexdigest()==g["packed_sha256"]
        meta=roster[g["id"]]; names=meta.get("models",meta.get("seat_names"))
        pair=set(names)=={"claude-AlphaCarbon-Master5s","claude-AlphaCarbon-Master6"}
        r=analyze(meta,json.loads(gzip.decompress(data)),pair)
        games.append(r)
        if pair:
            top.append(r["id"])
    assert len(top)==2
    report=dict(source_index_sha256=sha256(idx_bytes).hexdigest(),script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                definition="Largest absolute ten-transition change in seat-0 public cash lead, using integer cents. Ties choose earliest; direction identifies beneficiary irrespective of winner. Window [s,s+10] spans actions at s through s+9. Threat flags use the unit's current square at that time.",
                games=len(games),transitions=sum(g["end_step"] for g in games),direct_top_two=top,rows=games)
    args.out.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")


if __name__=="__main__":
    main()
