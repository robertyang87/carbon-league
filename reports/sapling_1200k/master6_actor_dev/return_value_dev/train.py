"""Estimate future enemy tax exposure from historical tree survival; no terminal scores are inputs."""
from collections import defaultdict,Counter
import json,gzip,sys
from pathlib import Path
from protocol import HERE,sha
from route import tree_key
TR=HERE.parent/"tree_response_training"
ROOT=HERE.parents[3]
sys.path.insert(0,str(TR/"planter_survival"))
from extract import FastCarbon

def games(split):
    p=json.loads((HERE/"protocol.json").read_bytes())
    for part in p[split]:
        if sha(ROOT/part["receipt"])!=part["sha256"]:raise ValueError("changed receipt")
        for game in part["games"]:
            path=ROOT/game["raw_path"]
            if sha(path)!=game["raw_sha256"]:raise ValueError("changed raw")
            yield game,json.loads(gzip.decompress(path.read_bytes()))

def fit():
    stats=defaultdict(Counter)
    for g,raw in games("train"):
        sim=FastCarbon(raw["game_configuration"]).reset(seed=g["seed"]); cfg=sim.config
        states=[]
        p=1-g["seat"]
        for j in raw["actions"]:
            pl=sim.full_observation()["players"]
            states.append((pl[p][3], [x for x,_,k in pl[1-p][2].values() if k=="PLANTER"]))
            sim.step(*j)
        states.append((sim.full_observation()["players"][p][3],None))
        for t,(trees,planters) in enumerate(states[:-1]):
            for uid,(pos,age) in trees.items():
                for h in (1,2,3):
                    if t+h>=len(states) or age+h>cfg["treeLifespan"]:continue
                    key=tree_key(pos,planters,h)
                    stats[g["seed"]][h,"n",key]+=1
                    stats[g["seed"]][h,"y",key]+=uid in states[t+h][0]
    obj=dict(p={},base={},training_maps=sorted(stats),horizons={})
    for h in (1,2,3):
        gamesizes={g:sum(n for (hh,k,_),n in cnt.items() if hh==h and k=="n") for g,cnt in stats.items()}
        total=sum(gamesizes.values())
        avg=total/len(stats)
        keys=[f"{h}/{j}" for j in range(5)]
        rates={key:[sum(stats[g][h,k,key]*avg/n for g,n in gamesizes.items()) for k in ("y","n")] for key in keys}
        mu=sum(n for n,d in rates.values())/total
        obj["base"][str(h)]=mu
        obj["p"].update({key:(a+20*mu)/(b+20) for key,(a,b) in rates.items()})
        obj["horizons"][str(h)]={"at_risk":total,"per_map":gamesizes}
    return obj
if __name__=="__main__":
    obj=fit()
    (HERE/"weights.json").write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
