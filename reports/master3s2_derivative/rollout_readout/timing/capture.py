"""Collect actual tensors from old training roots using the unmodified six-block teacher and freeze the replay panel."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[1]/"opponent_policy"
sys.path[:0]=[str(DATA/"search_probe"),str(DATA)]
from full_search_check import root_call


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def setup():
    plan=json.loads((HERE/"protocol.json").read_bytes())
    os.sched_setaffinity(0,set(plan["affinity"]))
    for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[k]="1"
    os.environ["CARBON_DEBUG"]="1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from original_policy import module, original
    raw=original()
    assert sha(raw)==plan["source_sha256"]
    mod=module(raw)
    # The signed module sets two threads when imported.
    torch.set_num_threads(1)
    assert mod._SEARCH_OK and mod._S_ABORT==.58 and mod._RNET is mod._NET and type(mod._net()).__name__=="_WinoNet"
    return plan,mod


def capture():
    plan,model=setup()
    import numpy as np
    from original_policy import PROTOCOL,free_entry
    old=(DATA/"compute/measurements.json.gz").read_bytes()
    roots=json.loads(gzip.decompress(old))["roots"]
    assert len(roots)==12
    source={ (c["terrain"],c["expert_seat"]):c for c in PROTOCOL["games"] if c["split"]=="train"}
    kind_of=type(model._NET)
    native=kind_of.__call__
    arrays={}
    cases=[]
    schedules={}
    for r in roots:
        schedules.setdefault((r["terrain"],r["seat"]),{})[r["t"]]=r
    for (terrain,seat),schedule in schedules.items():
        card=source[terrain,seat]
        tracker=model.BeliefTracker(model._Bel,seat)
        previous=None
        for t,obs,cfg,order in free_entry.public_history(card):
            if t in schedule:
                row=schedule[t]
                captured=[]
                def collect(self,x,sid,sc=None):
                    fn=sys._getframe(1)
                    origin=fn.f_code.co_name
                    kind={"_search":"root","other":"other","look":"joint" if fn.f_locals.get("both") else "own"}[origin]
                    start=time.perf_counter_ns()
                    ret=native(self,x,sid,sc)
                    captured.append((kind,x,sid,sc,time.perf_counter_ns()-start))
                    return ret
                kind_of.__call__=collect
                try:
                    pubcfg={k:v for k,v in cfg.items() if k!="randomSeed"}
                    trace=root_call(model,tracker,previous,obs,pubcfg,row["policy_rng"],row["starting_costs"])
                finally:
                    kind_of.__call__=native
                shapes=[]
                for j,(kind,x,sid,sc,elapsed) in enumerate(captured):
                    key=f"r{len(cases)}b{j}"
                    unit=np.rint(x*100).astype(np.uint8)
                    assert np.array_equal(x,unit.astype(np.float32)*np.float32(.01))
                    arrays[key]=unit
                    if sc is not None:arrays[key+"s"]=sc
                    shapes.append(dict(key=key,kind=kind,sid=int(sid),rows=len(x),scalar=sc is not None,original_live_forward_ns=elapsed))
                cases.append(dict(seed=card["seed"],terrain=terrain,seat=seat,t=t,receipt_sha256=card["sha256"],depth=trace["depth"],
                                  candidates=trace["candidates"],public_wall=trace["wall_s"],public_cpu=trace["cpu_s"],batches=shapes))
                print(card["seed"],seat,t,len(shapes),flush=True)
            tracker.step(obs["players"],t,obs["carbon"])
            tracker.set_actions(obs["players"],order)
            previous=obs["players"]
    with (HERE/"panel.npz").open("xb") as f:
        np.savez_compressed(f,**arrays)
    panel=dict(protocol_sha256=sha((HERE/"protocol.json").read_bytes()),incoming_costs_and_rng_sha256=sha(old),features_sha256=sha((HERE/"panel.npz").read_bytes()),public_roots=cases)
    with (HERE/"panel.json").open("x") as f:
        json.dump(panel,f,separators=(",",":"))
        f.write("\n")


if __name__=="__main__":
    capture()
