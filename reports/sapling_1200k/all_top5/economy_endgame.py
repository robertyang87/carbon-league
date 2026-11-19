"""Identify final-30 purchases, remaining living trees and cargo directly from archived observations."""
from collections import Counter
import gzip
import json


def last_assets(path):
    env=json.loads(gzip.decompress(path.read_bytes()))
    f=env["steps"];last=len(f)-1;start=max(0,last-30)
    cfg=env["configuration"];life=cfg["treeLifespan"]
    rows=[Counter(),Counter()]
    active={}
    def frame(j):
        ob=f[j][0]["observation"]
        trees={tid:(s,pos,age) for s,p in enumerate(ob["players"]) for tid,(pos,age) in p[3].items()}
        return ob,trees
    before,old=frame(start)
    for t in range(start+1,last+1):
        ob,cur=frame(t)
        for tid,entry in old.items():
            if tid not in cur:
                rows[entry[0]]["expired"]+=1
                if entry[2]<life:rows[entry[0]]["removed_before_lifespan"]+=1
                active.pop(tid,None)
        for tid,(s,pos,age) in cur.items():
            if tid not in old or old[tid][0]!=s:
                kind="plant" if tid not in old else "seize"
                rows[s][kind]+=1
                if t>last-5:rows[s]["final5_"+kind]+=1
                if t==last:rows[s]["final_step_"+kind]+=1
                active[tid]=s
            if tid in active and active[tid]==s:
                rows[s]["last30_new_tenure_absorption"]+=ob["trees"][tid][1]
        old=cur
    for s in (0,1):
        p=f[-1][0]["observation"]["players"][s]
        rows[s].update(final_trees=len(p[3]),final_worker_load=sum(w[1] for w in p[2].values()),
                       final_unexpired_tree_turns=sum(max(0,life-v[1]) for v in p[3].values()),
                       final_new_tenure_trees=sum(v==s for v in active.values()))
    return dict(scheduled_horizon=last==cfg["episodeSteps"]-1,time_window=[start,last],
                seats=[{k:round(v,3) for k,v in row.items()} for row in rows])
