"""Native public agent calls on previously declared seats and roots; identical RNG and initial cost per root."""
import copy,os,time
import numpy as np
from visibility import D,HERE,load,obs,read,report,data,sha
from build import build

def one(m,enabled,pre,prev,j,o,cfg,cal):
    if enabled is not None:m._MIXED_BELIEF=enabled
    tr=copy.deepcopy(pre)
    tr.__class__=m.BeliefTracker;tr.m=m._Bel;tr.bel.__class__=m.Belief
    assert type(tr.bel).estimate is m.Belief.estimate
    m._ST=dict(tr=tr,prev=prev,rng=np.random.RandomState(87391+j*31+o["step"]))
    m._S_COST[:]=cal;m._S_REC=[];m._S_STATS.pop("last",None)
    start=time.perf_counter()
    a=m.agent(o,cfg);elapsed=time.perf_counter()-start
    stats=m._S_STATS.get("last")
    if m._S_REC:
        _,_,menu,depth,score,ix=m._S_REC[-1]
        assert menu[ix]==a
        table=dict(menu=menu,depth=depth,score=score,index=ix)
    else:table=None
    return dict(orders=a,work=list(stats[:3]) if stats is not None else None,wall=elapsed,search=table)

def main():
    assert len(os.sched_getaffinity(0))==1
    raw=build();assert raw==(HERE/"candidate.py").read_bytes()
    p=load(D/"cold_start/candidate.py");m=load(HERE/"candidate.py")
    p._torch.set_num_threads(1)
    cards=read(D/"ladder/ds-pro-v1-placement.json")["games"]
    results=[]
    for j,card in enumerate(cards[:4]):
        tape,h=data(card);side=card["candidate_seat"];tr=p.BeliefTracker(p._Bel,side)
        cfg={k:v for k,v in tape["configuration"].items() if k!="randomSeed"}
        prev=None
        for t,frame in enumerate(tape["steps"][:71]):
            o=obs(frame,side)
            if t in (5,14,34,70):
                # Shared actual protocol calibration from the first three local full parent calls.
                seq=("P","E","E","D") if (j+(5,14,34,70).index(t))%2==0 else ("D","E","E","P")
                arms=[]
                for tag in seq:
                    a=one(p if tag=="P" else m, {"P":None,"E":True,"D":False}[tag],tr,prev,j,o,cfg,cal)
                    a["tag"]=tag;arms.append(a)
                pp=next(q for q in arms if q["tag"]=="P");dd=next(q for q in arms if q["tag"]=="D")
                ee=[q for q in arms if q["tag"]=="E"]
                def stable(a,b):
                    return a["work"] is not None and a["search"] is not None and a["work"]==b["work"] and a["orders"]==b["orders"] and a["search"]==b["search"]
                same=stable(pp,dd) and stable(*ee) and pp["work"]==ee[0]["work"]
                results.append(dict(card=card["match_id"],seat=side,step=t,replay_sha256=h,arms=arms,matched=same,
                                    attributable_changed=same and pp["orders"]!=ee[0]["orders"],untested_strength=True))
            if t<3:
                if t==0:p._ST={}
                p.agent(o,cfg)
                p._ST["tr"].set_actions(o["players"],tape["steps"][t+1][side]["action"] or {})
                if t==2:cal=list(p._S_COST)
            tr.step(o["players"],t,o["carbon"]);tr.set_actions(o["players"],tape["steps"][t+1][side]["action"] or {})
            prev=o["players"]
        print(card["match_id"],[(a["step"],a["matched"],a["attributable_changed"]) for a in results[-4:]],flush=True)
    report(HERE/"public_roots.json.gz",{"source":sha(raw),"records":results})
if __name__=="__main__":main()
