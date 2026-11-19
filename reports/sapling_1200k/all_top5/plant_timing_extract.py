"""Measure public planter choices and cash earned by each continuous ownership spell."""
from collections import defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
from economy_extract import FastCarbon,CashEvents,atomic,compact,distances

# Numeric rows keep the all-action evidence compact and have a versioned schema.
CHOICE=["root","seat","worker","cell","kind","price","neighbor_carbon","cash_after_price","remaining_turns",
        "old_tree_age","opp_planter_dist","own_collector_dist","opp_collector_dist","alt_adjacent_enemy_tree",
        "command","paid","next_same_worker_paid_in_5","other_cash_financed","own_trees","alive_global_trees","own_home_dist","opp_home_dist"]
PURCHASE=["root","seat","worker","tree_id","cell","kind","price","paid_turn","last_credited_turn","end_reason",
          "held_transitions","absorption","enemy_worker_absorption","collision_cash","payback_transitions","first_income_transitions"]
MOVES=("NONE","UP","RIGHT","DOWN","LEFT")

def version():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def extract(packed):
    env=json.loads(gzip.decompress(packed))
    cfg=env["configuration"];frames=env["steps"];T=len(frames)-1;size=cfg["size"];life=cfg["treeLifespan"]
    dist=distances(size)
    neigh=[]
    for i in range(size*size):
        row,col=divmod(i,size)
        neigh.append([((row-1)%size)*size+col,row*size+(col+1)%size,((row+1)%size)*size+col,row*size+(col-1)%size])
    sim=FastCarbon(cfg).load_observation(frames[0][0]["observation"]);sim.cash=CashEvents(sim.cash)
    active={};completed=[];choices=[];later=defaultdict(list);count_invalid=0
    def close(rec,j,reason):
        # start t, clearing transition t+1; purchase and absorption in same
        # transition can occur for a seizure. Expiry/ownership loss at j ends
        # the spell before its absorption phase; collision cash just before an
        # ownership change in j is still paid to the previous owner.
        rec[8]=j;rec[9]=reason
        rec[10]=j-rec[7]+(1 if reason=="terminal" else 0)
        completed.append(rec)
    for t in range(T):
        pre=frames[t];players=pre[0]["observation"]["players"]
        commands=[r["action"] or {} for r in frames[t+1]]
        trees={cell:(uid,seat,age) for seat,p in enumerate(players) for uid,(cell,age) in p[3].items()}
        alive=sum(v[2]<life for v in trees.values())
        fresh=cfg["plantCost"]+cfg["plantCostInflationRatio"]*cfg["plantCostInflationBase"]**alive
        selected={}
        for s in (0,1):
            own,opp=players[s],players[1-s];obs=pre[s]["observation"]["carbon"]
            bases=set(own[1].values())|set(opp[1].values())
            price=cfg["plantCost"] if len(own[3])<=cfg["playerTreeProtectiveNumber"] else fresh
            foe_planters=[w[0] for w in opp[2].values() if w[2]=="PLANTER"]
            our_collectors=[w[0] for w in own[2].values() if w[2]=="COLLECTOR"]
            foe_collectors=[w[0] for w in opp[2].values() if w[2]=="COLLECTOR"]
            for uid,(cell,load,role) in own[2].items():
                if role!="PLANTER" or cell in bases:continue
                tree=trees.get(cell)
                if tree and tree[1]==s and tree[2]<life:continue
                action=MOVES.index(commands[s].get(uid) or "NONE")
                kind="seize" if tree and tree[2]<life else "plant"
                cost=cfg["seizeCost"] if kind=="seize" else price
                if own[0]<cost and action:continue # a move cannot buy now
                # Runner knows its four cardinal neighbors. Full carbon is used
                # only by the offline accounting simulator, never by this choice row.
                carbon=sum(float(obs[j]) for j in neigh[cell] if j not in trees)
                d=dist[cell]
                rem=cfg["episodeSteps"]-1-t
                alt=sum(j in trees and trees[j][1]!=s and trees[j][2]<life-1 for j in neigh[cell])
                record=[t,s,uid,cell,kind,round(cost,4),round(carbon,2),round(own[0]-cost,3),rem,
                        tree[2] if tree else -1,min((d[v] for v in foe_planters),default=99),
                        min((d[v] for v in our_collectors),default=99),min((d[v] for v in foe_collectors),default=99),
                        alt,action,None,None,int(own[0]<cost),len(own[3]),alive,
                        min(d[v] for v in own[1].values()),min(d[v] for v in opp[1].values())]
                selected[s,uid]=record
        sim.cash.events.clear();sim.step(*commands)
        post=frames[t+1][0]["observation"]["players"]
        if any(abs(sim.cash[s]-post[s][0])>.011 for s in (0,1)):raise ValueError(f"cash mismatch t{t}")
        current={tree.pos:tid for tid,tree in sim.trees.items()}
        for k,s,val,*detail in sim.cash.events:
            if k in ("plant","seize"):
                uid,cell=detail; tid=current[cell];old=active.pop(tid,None)
                if old:close(old,t+1,"seized")
                root=selected.get((s,uid))
                if root is None:raise ValueError("actual purchase missing public pre-action choice")
                if root[4]!=k or abs(root[5]+val)>.005:raise ValueError("locked pre-price differs from booked cost")
                rec=[t,s,uid,tid,cell,k,round(-val,5),t+1,None,None,None,0.,0.,0.,None,None]
                active[tid]=rec;root[15]=len(later[s,uid]);later[s,uid].append(rec)
            elif k in ("tree_absorption","tree_collision_cash"):
                tid=detail[0] if k=="tree_absorption" else detail[2];rec=active.get(tid)
                if not rec:continue
                if rec[1]!=s:raise ValueError("receipt paid to different owner")
                rec[11 if k=="tree_absorption" else 13]+=val
                if k=="tree_absorption":rec[12]+=detail[2]
                turn=t+2-rec[7]
                if rec[15] is None and val>0:rec[15]=turn
                if rec[14] is None and rec[11]+rec[13]>=rec[6]-1e-7:rec[14]=turn
        for tid in list(active):
            if tid not in sim.trees:close(active.pop(tid),t+1,"expired")
        for rec in selected.values():
            if not rec[17] or rec[15] is not None:choices.append(rec)
    for rec in active.values():close(rec,T,"terminal")
    completed.sort(key=lambda x:(x[0],x[1],x[2]))
    by_key={(x[0],x[1],x[2]):i for i,x in enumerate(completed)}
    for c in choices:
        if c[15] is not None:c[15]=by_key[c[0],c[1],c[2]]
        if c[14]!=0:
            nex=next((r for r in later[c[1],c[2]] if c[0]<r[0]<=c[0]+5),None)
            if nex:c[16]=by_key[nex[0],nex[1],nex[2]]
    for r in completed:
        for i in (11,12,13):r[i]=round(r[i],5)
    if sum(c[15] is not None for c in choices)!=len(completed):raise ValueError("choice links lost")
    return dict(seed=cfg.get("randomSeed"),turns=T,rewards=env["rewards"],choices=choices,purchases=completed)

def main(manifest,raw,work):
    v=version();cards=json.loads(manifest.read_bytes())["games"];start=time.monotonic();cpu=time.process_time();n=0
    for game in cards:
        if not game.get("frame_verified"):continue
        gid=game["id"];p=work/(gid+".gz")
        if p.exists():
            old=json.loads(gzip.decompress(p.read_bytes()))
            if old["extractor_sha256"]==v and old["replay_sha256"]==game["sha256"]:continue
        packed=(raw/(gid+".json.gz")).read_bytes();sha=hashlib.sha256(packed).hexdigest()
        if sha!=game["sha256"]:raise ValueError(gid+" raw mismatch")
        r=extract(packed)
        if r["rewards"]!=game["score_by_seat"]:raise ValueError("score mismatch")
        doc=dict(id=gid,replay_sha256=sha,extractor_sha256=v,**r)
        atomic(p,gzip.compress(compact(doc),compresslevel=3,mtime=0))
        n+=1
        if n%80==0:print(n,gid,round((time.monotonic()-start)/n,3),flush=True)
    print("complete",n,"cpu",round(time.process_time()-cpu,2),"wall",round(time.monotonic()-start,2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","raw","work"):p.add_argument(name,type=Path)
    a=p.parse_args();main(a.manifest,a.raw,a.work)
