"""Run V5 and its NB12 arm on identical intact public observation histories on one CPU."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import tree_audited_game as arena

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WEIGHTS=ROOT/"data/weights/sapling_1200k"
ARMS={"v5":WEIGHTS/"soft_it50_v5_agent.py","deep":WEIGHTS/"soft_it50_deep214_agent.py"}
SEARCH=HERE/"tree_exam_soft_it50_v5"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(seconds):
    return dict(n=len(seconds),mean_ms=1000*statistics.mean(seconds),median_ms=1000*statistics.median(seconds),
                p95_ms=1000*float(np.quantile(seconds,.95)),max_ms=1000*max(seconds))


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"capacity_ab_inference.json")
    a=p.parse_args()
    if a.out.exists() or len(os.sched_getaffinity(0))!=1:
        raise ValueError("pin to one CPU and use a fresh output")
    games=json.loads((SEARCH/"results.json").read_text())["games"]
    games=sorted(games,key=lambda g:(g["seed"],g["candidate_seat"]))
    if [(g["seed"],g["candidate_seat"]) for g in games]!=[(s,p) for s in (620106811,620106812) for p in (0,1)]:
        raise ValueError("expected both seats on the frozen two development maps")
    arms={name:[] for name in ARMS}
    games_out=[]
    with threadpool_limits(limits=1):
        for i,g in enumerate(games):
            file=SEARCH/g["raw_file"]
            if sha(file)!=g["raw_sha256"]:
                raise ValueError("transcript hash changed")
            raw=json.loads(gzip.decompress(file.read_bytes()))
            cfg=raw["game_configuration"]
            side=g["candidate_seat"]
            sim=arena.exam.FastCarbon(cfg).reset(seed=g["seed"])
            frames=[]
            for step,actions in enumerate(raw["actions"]):
                obs=sim.obs(side)
                if obs["step"]!=step:
                    raise ValueError("time grid")
                frames.append(obs)
                sim.step(*actions)
            if arena.prior.hashed(sim.get_state())!=g["final_state_sha256"]:
                raise ValueError("terminal state did not match frozen game")
            order=("v5","deep") if i%2==0 else ("deep","v5")
            outputs={}
            local={}
            for model in order:
                path=ARMS[model]
                agent=arena.prior.shared.module(path,f"capacity_{model}_{i}")
                if agent._OPTS!=dict(teacher=2,shield=False,nocollide=False,endgame=False,pshield=False,rguard=False):
                    raise ValueError("changed policy wrapper")
                elapsed=[]
                decisions=[]
                for step,obs in enumerate(frames):
                    start=time.perf_counter_ns()
                    result=agent.agent(obs,cfg)
                    elapsed.append((time.perf_counter_ns()-start)/1e9)
                    if model=="v5" and result!=raw["actions"][step][side]:
                        raise ValueError(f"small agent did not reproduce its actual decision: {i}/{step}")
                    decisions.append(result)
                arms[model].extend(elapsed[1:])
                local[model]=dict(cold_first_call_ms=1000*elapsed[0],warm=summary(elapsed[1:]),
                                  warm_ns=[int(round(s*1e9)) for s in elapsed[1:]])
                outputs[model]=decisions
            changed=sum(s!=l for s,l in zip(outputs["v5"],outputs["deep"]))
            games_out.append(dict(map=g["seed"],seat=side,steps=len(frames),source_sha256=sha(file),
                                  order=order,public_action_maps_changed=changed,models=local))
        pools=[{k:p[k] for k in ("user_api","internal_api","num_threads")} for p in threadpool_info()]
    out=dict(meaning="Independent persistent public agents on identical complete dated V5 histories, source actions alone advance game. First call reported separately. Timing includes public agent and output, excludes simulator, archive replay, import and wire communication. Local shared CPU host.",
             cores=sorted(os.sched_getaffinity(0)),pools=pools,
             models={name:dict(pack_sha256=sha(path),pack_bytes=path.stat().st_size,warm=summary(arms[name]))
                     for name,path in ARMS.items()},games=games_out)
    with a.out.open("x") as f:
        json.dump(out,f,indent=2,allow_nan=False)
        f.write("\n")
    print({k:v["warm"] for k,v in out["models"].items()})


if __name__=="__main__":
    run()
