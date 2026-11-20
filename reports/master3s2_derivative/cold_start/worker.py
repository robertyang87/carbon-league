"""Fresh interpreter for a startup comparison; the plain arm runs byte-identical released source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import types

def profile_source(source):
    lines=source.decode().splitlines()
    res=[]
    for line in lines:
        i=len(line)-len(line.lstrip())
        indent=" "*i
        x=line.strip()
        before=after=None
        if x=="import numpy as np": before,after="numpy_import","numpy_import"
        if x=="_POLICY = _Policy(_WEIGHTS)": before,after="policy_decode","policy_decode"
        if x=="import torch as _torch":before="torch_import_and_threads"
        if x=="_torch.set_num_threads(2)":after="torch_import_and_threads"
        if x.startswith('_NET = _build(_POLICY,'):before,after="native_net_build_and_parity","native_net_build_and_parity"
        if x.startswith('for n in ') and ('the first pass of a new shape' in x or 'prime common sizes' in x):before="warm_sweep"
        if x.startswith('if _S_COST[0] == 0.0:'):
            res.append(indent+'_parts["warm_sweep"]=(_tick()-_t_warm_sweep)')
            before="original_cost_estimator"
        if x.startswith('_S_COST[1] = _S_COST[3] = 0.002'):
            after="original_cost_estimator"
        if before:res.append(indent+"_t_"+before+"=_tick()")
        res.append(line)
        if after:res.append(indent+'_parts["'+after+'"]=_tick()-_t_'+after)
    return "\n".join(res)+"\n"

def reply(d):
    print(json.dumps(d,separators=(",",":"),allow_nan=False),flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument("path",type=Path);a.add_argument("mode",choices=["plain","profile"]);args=a.parse_args()
    parts={}
    begin=time.perf_counter()
    blob=args.path.read_bytes()
    code=profile_source(blob) if args.mode=="profile" else blob
    t=time.perf_counter();obj=compile(code,str(args.path),"exec")
    compiled=time.perf_counter()-t
    mod=types.ModuleType("unmodified_cold_actor")
    mod.__dict__.update(_tick=time.perf_counter,_parts=parts)
    start=time.perf_counter()
    exec(obj,mod.__dict__)
    executed=time.perf_counter()-start
    assert mod._SEARCH_OK and mod._NET is not None and mod._RNET is mod._NET and mod._S_ABORT==.58 and mod._SEED is None
    reply(dict(ready=True,mode=args.mode,source=hashlib.sha256(blob).hexdigest(),compile_s=compiled,exec_s=executed,
        relative_ready_s=time.perf_counter()-begin,stages=parts,python=sys.version.split()[0],numpy=mod.np.__version__,
        torch=mod._torch.__version__,torch_threads=mod._torch.get_num_threads(),cost=mod._S_COST.copy(),net=type(mod._NET).__name__,
        rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        mem_max=Path("/sys/fs/cgroup/memory.max").read_text().strip(),affinity=sorted(os.sched_getaffinity(0))))
    for line in sys.stdin:
        req=json.loads(line);op=req["op"]
        if op=="quit":break
        if op=="restore":
            me=req["seat"];hist=req["history"]
            tracker=mod.BeliefTracker(mod._Bel,me)
            prev=None
            for row in hist:
                obs=row["observation"];p=obs["players"]
                tracker.step(p,obs["step"],obs["carbon"]);tracker.set_actions(p,row["wire"]);prev=p
            mod._ST=dict(tr=tracker,prev=prev,rng=mod.np.random.RandomState(req["sample_seed"]))
            reply(dict(restored=len(hist)));continue
        if op=="correct":
            mod._ST["tr"].set_actions(req["players"],req["wire"]);reply(dict(ok=True));continue
        obs=req["observation"]
        if obs["step"]==0: mod._SEED=req["sample_seed"]
        mod._S_STATS.pop("last",None)
        mod._S_REC=[]
        initial_abort=mod._S_STATS["aborted"]
        t=time.perf_counter(); action=mod.agent(obs,req["configuration"]);duration=time.perf_counter()-t
        mod._SEED=None
        last=mod._S_STATS.get("last")
        reply(dict(action=action,step=obs["step"],agent_only_s=duration,search_last=last,
            search_recorded=bool(mod._S_REC),aborts=mod._S_STATS["aborted"]-initial_abort,
            rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

if __name__=="__main__":main()
