"""Replay each available Top Five scorecard once and publish resumable economic evidence."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from statistics import median

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from turning_point_ledger import CashEvents, FastCarbon

KEYS=("bank","donation","tree","collision_cash","recruit","plant","seize","round")
POS={"own_delivery":0,"opponent_worker_delivery_to_this_base":1,"tree_absorption":2,
     "base_collision_cash":3,"tree_collision_cash":3,"hire_collector":4,"hire_planter":4,
     "plant":5,"seize":6,"rounding":7}
BASELINE_TOP5=("claude-AlphaCarbon-Master5s","claude-AlphaCarbon-Master6","RyanX-AIPowercore-it720-preview",
               "RyanX-AIPowercore-M9XJ-630","ClaudeFable5.2-v7")
DIST={}
def distances(size):
    if size not in DIST:
        def dist(i,j):
            dr=abs(i//size-j//size);dc=abs(i%size-j%size)
            return min(dr,size-dr)+min(dc,size-dc)
        DIST[size]=[[dist(i,j) for j in range(size*size)] for i in range(size*size)]
    return DIST[size]

def digest(data):return hashlib.sha256(data).hexdigest()
def compact(o):return json.dumps(o,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    tmp.write_bytes(data);os.replace(tmp,path)
def cents(x):return round(x,2)


def visible_ledger(frames,cfg):
    """Independent public action/roster fallback when archived map cannot be exactly reset."""
    sys.path.insert(0,str(HERE.parents[1]/"master_family/deep_review"))
    from carbon_deep_events import _actions,_snapshot,_reconstruct
    prev=_snapshot(frames[0],cfg["size"])
    timeline=[];drops=[];banks=[];cast=[]
    for j in range(1,len(frames)):
        nxt=_snapshot(frames[j],cfg["size"]);actions=_actions(frames[j])
        totals=[[0.]*8,[0.]*8];loots=[0.,0.];trees_paid=[None,None];tx=[]
        for s in (0,1):
            ev=_reconstruct(prev,nxt,actions,cfg,s)
            for e in ev:
                k=e["kind"]
                if k in ("delivery","enemy_delivery_to_us"):
                    totals[s][0 if k=="delivery" else 1]+=e["net_cash"]
                    if k=="delivery":tx.append((s,e["worker_id"],e["net_cash"]))
                elif k=="collision":
                    loots[s]+=e["enemy_cargo_captured"]
                    if e["recipient"] in ("tree_cash","base_cash") and e["recipient_seat"]==s:
                        totals[s][3]+=sum(w["known_cargo"] for w in e["participants"] if w["worker_id"] in e["deleted_worker_ids"])
                elif k=="recruit":totals[s][4]-=e["cost"]
                elif k=="tree_captured":totals[s][6]-=cfg["seizeCost"]
            obs=frames[j][0]["observation"]
            totals[s][2]=sum(obs["trees"][uid][1] for uid in obs["players"][s][3])
            # Public residual includes planting charges and treasury rounding. Keep
            # them combined in fallback records rather than claiming exact planting.
            totals[s][5]=nxt["cash"][s]-prev["cash"][s]-sum(totals[s])
        timeline.append(totals);drops.append(loots);banks.append(tx);cast.append(trees_paid);prev=nxt
    return timeline,drops,banks,cast


def reconstruct(frames,cfg):
    sim=FastCarbon(cfg).load_observation(frames[0][0]["observation"]);sim.cash=CashEvents(sim.cash)
    timeline=[];loots=[];banks=[];tax=[]
    for j,frame in enumerate(frames[1:],1):
        acts=[v["action"] or {} for v in frame]
        sim.cash.events.clear();sim.step(*acts)
        players=frame[0]["observation"]["players"]
        # Verify both cumulative balances and every public worker ID, position,
        # cargo and tree owner/age. No policy or new games are run here.
        for s in (0,1):
            if abs(sim.cash[s]-players[s][0])>.011 or len(sim.player_workers[s])!=len(players[s][2]) or len(sim.player_trees[s])!=len(players[s][3]):
                raise ValueError(f"cash/roster mismatch at t{j}, seat{s}")
            for uid,v in players[s][2].items():
                w=sim.workers.get(uid)
                if w is None or w.pos!=v[0] or w.player!=s or abs(w.carbon-v[1])>.015 or w.is_collector!=(v[2]=="COLLECTOR"):
                    raise ValueError(f"worker state mismatch at t{j}")
            for tid,v in players[s][3].items():
                tr=sim.trees.get(tid)
                if tr is None or tr.pos!=v[0] or tr.player!=s or tr.age!=v[1]:
                    raise ValueError(f"tree state mismatch at t{j}")
        amount=[[0.]*8,[0.]*8];transfers=[];enemy_fees=[0.,0.];taken=[0.,0.]
        for kind,s,val,*detail in sim.cash.events:
            amount[s][POS[kind]]+=val
            if kind=="own_delivery" and val>0:transfers.append((s,detail[0],val))
            if kind=="tree_absorption":enemy_fees[s]+=detail[2]
        for uid,s,val,recipient,_ in sim.last_collisions:
            if recipient==1-s:taken[recipient]+=val
        timeline.append(amount);banks.append(transfers);tax.append(enemy_fees);loots.append(taken)
    return timeline,loots,banks,tax


def profile(players,s,dist):
    p,opp=players[s],players[1-s]
    homes=list(p[1].values())
    foes=[(w[0],w[1]) for w in opp[2].values() if w[2]=="COLLECTOR"]
    lifts=[]
    for uid,(cell,load,role) in p[2].items():
        if role!="COLLECTOR" or load<=0:continue
        near=min(dist[cell][h] for h in homes)
        contested=any(w<=load and dist[cell][other]<=1 for other,w in foes)
        lifts.append((uid,cell,load,near,contested))
    heaviest=max(lifts,key=lambda v:v[2]) if lifts else None
    return dict(cash=p[0],workers=len(p[2]),collectors=sum(w[2]=="COLLECTOR" for w in p[2].values()),
             trees=len(p[3]),trees_age40plus=sum(v[1]>=40 for v in p[3].values()),
             cargo=cents(sum(v[2] for v in lifts)),safe_close_load=cents(sum(v[2] for v in lifts if v[3]<=3 and not v[4])),
             heaviest=[heaviest[0],heaviest[1],cents(heaviest[2]),heaviest[3],heaviest[4]] if heaviest else None)


def summarize(frames,ledgers,loots,banks,tax,cfg,rewards):
    n=len(ledgers);dist=distances(cfg["size"])
    ps=[row[0]["observation"]["players"] for row in frames]
    margin=[p[0][0]-p[1][0] for p in ps]
    W=min(10,n)
    start=max(range(n-W+1),key=lambda t:abs(margin[t+W]-margin[t]));end=start+W
    win=0 if rewards[0]>rewards[1] else 1 if rewards[1]>rewards[0] else None
    persistent=None
    if win is not None and (margin[-1] if win==0 else -margin[-1])>0:
        last=max((t for t,v in enumerate(margin) if (v if win==0 else -v)<=0),default=-1)
        persistent=last+1
    def window(a,b):
        categories=[]
        for s in (0,1):
            arr={k:cents(sum(x[s][v] for x in ledgers[a:b])) for v,k in enumerate(KEYS)}
            gross=cents(sum(x[s] for x in loots[a:b]))
            amounts=[e for row in banks[a:b] for e in row if e[0]==s]
            tariff=cents(sum(x[s] for x in tax[a:b])) if all(x[s] is not None for x in tax[a:b]) else None
            arr.update(cargo_acquired_from_enemy=gross,tree_from_enemy_carried=tariff,bank_count=len(amounts),
                       bank_median=cents(median([e[2] for e in amounts])) if amounts else 0.)
            exact=ps[b][s][0]-ps[a][s][0]
            if abs(sum(arr[k] for k in KEYS)-exact)>.06:raise ValueError("interval cash does not reconcile")
            categories.append(arr)
        return dict(a=a,b=b,margin0_delta=cents(margin[b]-margin[a]),seats=categories)
    def detail(a,b):
        result=window(a,b)
        result["positions"]=[dict(t=j,seats=[profile(ps[j],s,dist) for s in (0,1)])
                             for j in sorted({max(0,a-k) for k in (10,5,3,0)}|{b})]
        hits=[[] for _ in range(2)]
        for j in range(a,b):
            for s,uid,amount in banks[j]:
                hits[s].append((amount,j+1,uid))
        result["largest_receipt_paths"]=[]
        for s in (0,1):
            if not hits[s]:result["largest_receipt_paths"].append(None);continue
            amount,j,uid=max(hits[s]);path=[]
            for t in sorted({max(0,j-k) for k in (10,5,3,1,0)}):
                w=ps[t][s][2].get(uid);home=ps[t][s][1].values()
                path.append([t,w[0],cents(w[1]),min(dist[w[0]][h] for h in home)] if w else [t,None,None,None])
            result["largest_receipt_paths"].append(dict(t=j,uid=uid,amount=cents(amount),path=path))
        return result
    alerts=[{"approaching":[0,0],"other":[0,0]} for s in (0,1)]
    by_uid=defaultdict(list)
    for i,tx in enumerate(banks,1):
        for s,uid,val in tx:by_uid[s,uid].append(i)
    for s in (0,1):
        last={}
        for t in range(3,n-9):
            p=ps[t][s];homes=list(p[1].values())
            for uid,(cell,amount,role) in p[2].items():
                if role!="COLLECTOR" or amount<200 or t-last.get(uid,-20)<10:continue
                d=min(dist[cell][h] for h in homes)
                old=ps[t-3][s][2].get(uid)
                if not 4<=d<=8 or old is None:continue
                last[uid]=t
                tag="approaching" if d<min(dist[old[0]][h] for h in homes) else "other"
                alerts[s][tag][1]+=1;alerts[s][tag][0]+=any(t<j<=t+10 for j in by_uid[s,uid])
    spans=[(a,min(a+100,n)) for a in range(0,n,100)]
    return dict(turns=n,rewards=rewards,cash_end=[p[0] for p in ps[-1]],winner=win,whole=window(0,n),
                stages=[window(a,b) for a,b in spans],last30=window(max(0,n-30),n),
                swing=detail(start,end),persistent_cash_lead_t=persistent,
                decisive=detail(max(0,persistent-10),persistent) if persistent is not None else None,alerts=alerts)


def analyze(raw):
    env=json.loads(gzip.decompress(raw));frames=env["steps"];cfg=env["configuration"];rewards=env["rewards"]
    if len(frames)<2 or len(rewards)!=2 or not all(isinstance(x,(int,float)) for x in rewards):
        return dict(method="unscored_replay",reason="missing transitions or numeric rewards",turns=len(frames)-1,rewards=rewards)
    if [f[0]["observation"]["step"] for f in frames]!=list(range(len(frames))):raise ValueError("nonconsecutive roots")
    try:
        entries=reconstruct(frames,cfg);method="simulator_verified";reason=None
    except (ValueError,KeyError,TypeError,IndexError) as exc:
        reason=str(exc);entries=visible_ledger(frames,cfg);method="public_roster_verified"
    summary=summarize(frames,*entries,cfg,rewards)
    return dict(method=method,fallback_reason=reason,statuses=env["statuses"],seed=cfg.get("randomSeed"),**summary)


def process(cache,cards,work,limit=0):
    identity=digest(Path(__file__).read_bytes())
    count=0;bad=[];seen=0;started=time.monotonic();elapsed=0.
    for entry in sorted(cards,key=lambda r:r["id"]):
        gid=entry["id"];rawpath=cache/(gid+".json.gz")
        if not rawpath.exists():continue
        seen+=1
        dest=work/(gid+".json.gz")
        if dest.exists():
            old=json.loads(gzip.decompress(dest.read_bytes()))
            if old["extractor_sha256"]==identity and (not entry.get("sha256") or old["replay_sha256"]==entry["sha256"]):
                continue
        raw=rawpath.read_bytes();start=time.monotonic();cpu=time.process_time()
        try:
            if entry.get("sha256") and digest(raw)!=entry["sha256"]:raise ValueError("sealed gzip identity mismatch")
            record=analyze(raw)
            if record.get("rewards")!=entry["score_by_seat"]:raise ValueError("replay score differs from scorecard")
        except Exception as e:
            bad.append((gid,type(e).__name__,str(e)))
            continue
        duration=time.monotonic()-start
        doc=dict(id=gid,replay_sha256=digest(raw),extractor_sha256=identity,cpu_seconds=round(time.process_time()-cpu,6),record=record)
        atomic(dest,gzip.compress(compact(doc),compresslevel=4,mtime=0))
        elapsed+=duration;count+=1
        if count%50==0:print("saved",count,"last",gid,"s/game",round(elapsed/count,3),flush=True)
        if limit and count>=limit:break
    print("saved",count,"available",seen,"errors",len(bad),"wall",round(time.monotonic()-started,2),"s/game",round(elapsed/max(count,1),3),flush=True)
    if bad: print(bad[:12],flush=True)
    return count,seen,bad


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest",type=Path);p.add_argument("cache",type=Path);p.add_argument("work",type=Path);p.add_argument("--limit",type=int,default=0)
    a=p.parse_args();manifest=json.loads(a.manifest.read_bytes());process(a.cache,manifest["games"],a.work,a.limit)
