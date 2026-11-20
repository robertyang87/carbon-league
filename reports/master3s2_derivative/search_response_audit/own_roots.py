"""Study only the original examiner's root proposal on seven published H2H games."""
import argparse
import copy
from collections import defaultdict,Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/"freight_forecast"),str(HERE.parent/"opponent_policy")]
import dev_tapes as tapes
from original_policy import original,module,sha
STEPS=(0,20,72,215,267,285,296,298)


class Observer:
    def __init__(self,est,known):
        self.est,self.known=est,known
    def step(self,*unused):
        return self.est,self.known
    def set_actions(self,*unused):
        pass


def snap(m,obs,est,known,previous,actual,seed,ordinal):
    t=obs["step"];seat=obs["player"];p=obs["players"][seat]
    x=m._quant(m.encode(obs["players"],t,seat,est,known,previous))
    native=m._POLICY.forward
    lw,lb=native(x,m._SID)
    m._POLICY.forward=lambda _x,_sid:(lw,lb)
    def act(temp,rngseed):
        m._TEMP=temp
        m._ST=dict(tr=Observer(est,known),prev=previous,rng=np.random.RandomState(rngseed))
        if t==0:
            # Original _act forcibly resets tracker and game RNG at t0. Match the entered sampled seed.
            m._SEED=rngseed
        return m._act(obs,None)
    try:
        greedy=act(0.,329)
        draws=[act(1.,30119+17*ordinal+k) for k in range(32)]
        m._TEMP=1.
        ent={};stats={}
        for units,kind,logits,n in ((p[1],"base",lb,3),(p[2],"worker",lw,5)):
            keys=list(units)
            off_mode=0.;hn=0.
            for uid in keys:
                offset=units[uid] if kind=="base" else units[uid][0]
                pr=m._probs(logits[offset])
                hn+=float(-sum(float(v)*np.log(float(v)) for v in pr if v>0))
                off_mode+=1-float(max(pr))
            stats[kind]=dict(units=len(keys),n=len(keys)*len(draws),entropy_sum=hn,off_mode_sum=off_mode,
                              issued_greedy_diffs=sum(actual.get(u)!=greedy.get(u) for u in keys),
                              t1_vs_greedy=sum(d.get(u)!=greedy.get(u) for d in draws for u in keys))
        return dict(t=t,seed=seed,original_seat=seat,raw_proposal=stats,draws=len(draws),
                    joint_t1_vs_greedy=sum(d!=greedy for d in draws),issued_vs_greedy=actual!=greedy)
    finally:
        m._TEMP=1.;m._SEED=None;m._POLICY.forward=native

def run(source):
    m=module(source)
    groups=defaultdict(Counter);rows=[];skipped=[]
    for card,path,_ in tapes.inventory():
        seat=1-card["seat"]
        if card["status"]!="complete":
            skipped.append(dict(seed=card["seed"],original_seat=seat,status=card["status"]));continue
        results=card["official_scores_by_seat"];outcome="W" if results[seat]>results[1-seat] else "L" if results[seat]<results[1-seat] else "D"
        tr=m.BeliefTracker(m._Bel,seat);previous=None
        for info in tapes.rows(path,seat):
            obs=info["obs"];step=info["t"];est,known=tr.step(obs["players"],step,obs["carbon"])
            if step in STEPS:
                row=snap(m,obs,est,known,previous,info["own"],card["seed"],len(rows))
                row["original_outcome"]=outcome;rows.append(row)
                phase="opening" if step==0 else "middle" if step in (20,72,215) else "late"
                for label in (phase,phase+"_"+outcome,"all","all_"+outcome):
                    g=groups[label];g.update({"roots":1,"draws":row["draws"],"joint_changed":row["joint_t1_vs_greedy"],
                                               "final_issued_greedy_diff":int(row["issued_vs_greedy"])})
                    for typ,values in row["raw_proposal"].items():
                        g.update({typ+"_"+k:v for k,v in values.items()})
            tr.set_actions(obs["players"],info["own"]);previous=obs["players"]
    result=dict(source=sha(source),hosted_settlement=tapes.SETTLEMENT,
                steps=STEPS,rows=rows,groups=groups,skipped=skipped)
    raw=json.dumps(result,separators=(",",":"),allow_nan=False).encode()
    (HERE/"own_roots.json.gz").write_bytes(gzip.compress(raw,mtime=0))
    for k in sorted(groups):
        g=groups[k];print(k,g["roots"],g["joint_changed"]/g["draws"],g["worker_t1_vs_greedy"]/g["worker_n"] if g["worker_n"] else None,g["final_issued_greedy_diff"],flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--original",type=Path);a=ap.parse_args()
    source=a.original.read_bytes() if a.original else original()
    assert sha(source)==tapes.ORIGINAL
    run(source)
