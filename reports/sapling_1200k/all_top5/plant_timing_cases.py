"""Run conditional move-vs-purchase terminal branches and trace representative earlier public observations."""
from collections import Counter,defaultdict
from pathlib import Path
import gzip
import json
import numpy as np

from economy_extract import FastCarbon,distances
from plant_timing_extract import MOVES


def public_path(env,c):
    t,s,uid,cell=c[:4];size=env["configuration"]["size"];dist=distances(size);rows=[]
    for j in sorted({max(0,t-lag) for lag in (10,5,3,0)}):
        frame=env["steps"][j];obs=frame[0]["observation"];own,opp=obs["players"][s],obs["players"][1-s]
        worker=own[2].get(uid);r=dict(t=j,cash=round(own[0],2),trees=len(own[3]),opposing_trees=len(opp[3]),unit_present=worker is not None)
        if worker:
            where=worker[0];ds=dist[where];row,col=divmod(where,size)
            neigh=[((row-1)%size)*size+col,row*size+(col+1)%size,((row+1)%size)*size+col,row*size+(col-1)%size]
            occupied={v[0] for p in (own,opp) for v in p[3].values()}
            carbon=frame[s]["observation"]["carbon"]
            r.update(cell=where,relative_to_event_cell=ds[cell],seen_adjacent_carbon=round(sum(carbon[n] for n in neigh if n not in occupied),2),
                     nearest_enemy_planter=min((ds[w[0]] for w in opp[2].values() if w[2]=="PLANTER"),default=None),
                     nearest_friendly_collector=min((ds[w[0]] for w in own[2].values() if w[2]=="COLLECTOR"),default=None))
        rows.append(r)
    return rows


def benchmark(env,choices):
    """Root alternatives selected without future rival orders; all other units replay recorded actions."""
    cfg=env["configuration"];steps=env["steps"];T=len(steps)-1;D=distances(cfg["size"])
    sims=FastCarbon(cfg).load_observation(steps[0][0]["observation"]);out=[]
    by_t=defaultdict(list)
    for c in choices:by_t[c[0]].append(c)
    for t in range(max(by_t)+1):
        joint=[v["action"] or {} for v in steps[t+1]]
        if t in by_t:
            saved=sims.get_state()
            for c in by_t[t]:
                s,uid,cell=c[1:4];st=steps[t][0]["observation"]["players"]
                their=st[1-s];ours=st[s];home=list(ours[1].values())
                # Root danger uses only current reachable zones, our own simultaneously
                # planned actions, and nearest own bank as stable tie-breaks.
                def rank(a):
                    dest=sims._move_nbr[a][cell]
                    enemy=any(D[dest][w[0]]<=1 for w in their[2].values())
                    ally=any(v[0]==dest for u,v in ours[2].items() if u!=uid)
                    return (dest in their[1].values(),enemy,ally,min(D[dest][b] for b in home),MOVES.index(a))
                ranking=sorted(MOVES[1:],key=rank)
                arms=[]
                for first in ("NONE",*ranking):
                    sim=FastCarbon(cfg).set_state(saved)
                    early=False
                    for j in range(t,T):
                        orders=[dict(v["action"] or {}) for v in steps[j+1]]
                        if j==t:
                            if first=="NONE":orders[s].pop(uid,None)
                            else:orders[s][uid]=first
                        sim.step(*orders)
                        if j+1<T and any(not sim.player_workers[pl] and (not sim.bases[pl] or sim.cash[pl]<min(cfg["recCollectorCost"],cfg["recPlanterCost"])) for pl in (0,1)):
                            early=True;break
                    arms.append(dict(first=first,cash=[round(float(x),2) for x in sim.cash],early=early))
                baseline=arms[0]
                if baseline["cash"]!=[p[0] for p in steps[-1][0]["observation"]["players"]] or baseline["early"]:
                    raise ValueError("actual purchase failed final-state ledger comparison")
                delta=None if arms[1]["early"] else round((arms[1]["cash"][s]-arms[1]["cash"][1-s])-
                                                        (baseline["cash"][s]-baseline["cash"][1-s]),2)
                out.append(dict(t=t,seat=s,worker=uid,cell=cell,purchase_kind=c[4],price=c[5],preselected_move=ranking[0],arms=arms,
                                margin_delta=delta,public_path=public_path(env,c)))
        sims.step(*joint)
    return out


def case_summary(c,rec=None):
    t,s,uid,cell,kind,price,ncarbon,reserve,left,age,enemy,guard=c[:12]
    result=dict(t=t,seat=s,unit=uid,cell=cell,kind=kind,price=price,visible_c4=ncarbon,working_cash=reserveround(reserve),left=left,
                tree_age=age,enemy_planter_distance=enemy,friendly_collector_distance=guard,logged_command=MOVES[c[14]])
    if rec is not None:
        result.update(eventual_purchase_root=rec[0],eventual_purchase_kind=rec[5],paid=rec[6],absorbed=rec[11],collision_cash=rec[13],held=rec[10],
                      net=round(rec[11]+rec[13]-rec[6],2),payback=rec[14],end_reason=rec[9])
    return result


def reserveround(x):
    return round(x,2)
