"""Exercise the existing optional second rollout on preselected training roots at the native wall budget."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "cold_start/candidate.py"
UPSTREAM = HERE.parent / "opponent_policy"
sys.path.insert(0, str(UPSTREAM))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def capture_source(raw):
    code = raw.decode().replace("\r\n", "\n")
    updates = {
        "    while not cut and depth == D2": "    first_finalists = tuple(alive)\n    while not cut and depth == D2",
        "    if depth == 0:\n        _S_STATS": (
            "    _S_STATS['_diagnostic'] = dict(menu=cands, first=first_finalists, final=tuple(alive),"
            " values=dict(val), opponent=dict(opp), started='new' in locals(), cut=cut, rounds=rolls,"
            " depth=depth)\n    if depth == 0:\n        _S_STATS"
        ),
    }
    for old, new in updates.items():
        if code.count(old) != 1:
            raise ValueError("Search diagnostics no longer apply uniquely")
        code = code.replace(old, new)
    return code.encode()


def call(mod, tracker, previous, obs, cfg, seed, costs, cap):
    import numpy as np
    mod._ST = dict(tr=copy.deepcopy(tracker), prev=previous, rng=np.random.RandomState(seed))
    mod._S_COST[:] = costs
    mod._S_STATS.pop("_diagnostic", None)
    mod._S_STATS.pop("last", None)
    mod._S_MMAX = cap
    begin, cpu = time.perf_counter(), time.process_time()
    issued = mod.agent(obs, cfg)
    elapsed, consumed = time.perf_counter()-begin, time.process_time()-cpu
    trace = mod._S_STATS.get("_diagnostic")
    row = dict(issued=issued, wall_s=elapsed, cpu_s=consumed, ending_cost=list(mod._S_COST))
    if trace is None:
        row.update(menu=[issued], depth=0, stage="single_or_no_search", rounds=0, finalists=[], first_values={}, second_values={})
        return row

    vals, first, alive = trace["values"], trace["first"], trace["final"]
    rolls, depth = trace["rounds"], trace["depth"]
    row.update(menu=trace["menu"], depth=depth, rounds=rolls, first_finalists=first, finalists=alive,
               first_values={str(c):float(vals[c,0]) for c in first if (c,0) in vals},
               second_values={str(c):float(vals[c,1]) for c in alive if (c,1) in vals},
               opponent_orders=trace["opponent"])
    if not depth:
        row["stage"]="zero_depth"
        return row

    def choose(candidates, rounds):
        mean={c:sum(vals[c,r] for r in range(rounds))/rounds for c in candidates}
        best=max(candidates,key=mean.get)
        return (best if mean[best]-mean[0]>mod._S_EPS else 0)
    pre = choose(first,1)
    post = choose(alive,rolls)
    assert issued == row["menu"][post]
    row.update(first_choice=pre, reconstructed_choice=post, within_call_changed=row["menu"][pre]!=issued)
    if cap == 1:
        stage="control"
    elif depth != mod._S_D2 or trace["cut"] or len(first)<2:
        stage="first_ineligible"
    elif rolls==2:
        stage="second_complete"
        assert tuple(alive)==(0,max((c for c in first if c),key=lambda c:vals[c,0]))
        assert len(row["second_values"])==2 and len(row["opponent_orders"])==2
    elif trace["started"]:
        stage="second_partial_dropped"
        assert not row["second_values"]
    else:
        stage="second_budget_denied"
    row["stage"]=stage
    return row


def main(core):
    import platform
    from collections import Counter
    os.sched_setaffinity(0,{core})
    for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[k]="1"
    os.environ["CARBON_DEBUG"]="1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from original_policy import PROTOCOL, module, free_entry
    plan_raw=(HERE/"protocol.json").read_bytes()
    plan=json.loads(plan_raw)
    raw=SOURCE.read_bytes()
    assert sha(raw)==plan["base_sha256"]
    assert sha((UPSTREAM/"protocol.json").read_bytes())==plan["frozen_source_protocol_sha256"]
    mod=module(capture_source(raw))
    assert mod._S_M==mod._S_MMAX==1 and mod._S_D2==8 and mod._S_ABORT==.58
    assert mod._TEMP==1. and mod._SEED is None and mod._net() is not None and mod._SEARCH_OK
    initial=list(mod._S_COST)
    rows=[]
    for c in sorted(PROTOCOL["games"],key=lambda c:(c["terrain"],c["expert_seat"])):
        seat=c["expert_seat"]
        selected={r["step"]:r for r in plan["roots"] if (r["source"],r["seat"])==(c["path"],seat)}
        if not selected:continue
        root=HERE.parents[2]
        assert c["split"]=="train" and sha((root/c["path"]).read_bytes())==c["sha256"]
        tr=mod.BeliefTracker(mod._Bel,seat)
        prev=None
        for t,obs,cfg,archive in free_entry.public_history(c):
            target=next((s for s in selected if s-3<=t<=s),None)
            if target is not None:
                seed=int.from_bytes(hashlib.sha256(f'original-compute-v1/{c["terrain"]}/{seat}/{t}'.encode()).digest()[:4],"big")
                cfg={k:v for k,v in cfg.items() if k!="randomSeed"}
                assert "full_carbon" not in obs and obs["player"]==seat
                if t==target-3:cost=initial
                if t<target:
                    cost=call(mod,tr,prev,obs,cfg,seed,cost,1)["ending_cost"]
                else:
                    order=[1,2] if len(rows)%2==0 else [2,1]
                    arms={n:call(mod,tr,prev,obs,cfg,seed,cost,n) for n in order}
                    a,b=arms[1],arms[2]
                    same=all(a.get(k)==b.get(k) for k in ("menu","depth","first_finalists","first_values"))
                    if same and b.get("stage")=="second_complete":
                        assert (a["issued"]!=b["issued"])==b["within_call_changed"]
                    rows.append(dict(**selected[t],private_seed=seed,cost=cost,order=order,arms=arms,
                                     first_stage_exact=same,across_calls_changed=a["issued"]!=b["issued"]))
            tr.step(obs["players"],t,obs["carbon"])
            tr.set_actions(obs["players"],archive)
            prev=obs["players"]
        print(c["seed"],seat,len(rows),flush=True)
    assert len(rows)==len(plan["roots"])==12
    out=dict(base=sha(raw),protocol=sha(plan_raw),diagnostic_sha256=sha(Path(__file__).read_bytes()),
             traced_source_sha256=sha(capture_source(raw)),affinity=list(os.sched_getaffinity(0)),group_cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
             group_memory_max=Path("/sys/fs/cgroup/memory.max").read_text().strip(),torch_version=torch.__version__,threads=torch.get_num_threads(),cpu=platform.processor(),initial_cost=initial,
             stages=Counter(r["arms"][2]["stage"] for r in rows),
             second_changed=sum(bool(r["arms"][2].get("within_call_changed")) for r in rows),first_stage_exact=sum(bool(r["first_stage_exact"]) for r in rows),roots=rows)
    dest=HERE/"roots.json.gz"
    with dest.open("xb") as f:
        f.write(gzip.compress((json.dumps(out,sort_keys=True,allow_nan=False)+"\n").encode(),mtime=0))
    print(out["stages"],out["second_changed"],out["first_stage_exact"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--cpu",type=int,required=True)
    main(p.parse_args().cpu)
