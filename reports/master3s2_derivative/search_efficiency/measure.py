"""Compare the complete parent and memory-optimized search on the same recorded physical roots."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "opponent_policy"
PANEL = HERE.parent / "rollout_readout/timing"
sys.path[:0] = [str(PANEL), str(DATA / "search_probe"), str(DATA)]
from full_search_check import root_call
from original_policy import module, PROTOCOL, free_entry
from fixed_work import insert_hooks
from build import PARENT, build
import numpy as np
import torch


def sha(data):
    return hashlib.sha256(data).hexdigest()


def equal_tensors(left, right):
    return len(left) == len(right) and all(
        a is None and b is None or
        a is not None and b is not None and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        for a,b in zip(left,right))


def roots():
    schedule = json.loads((PANEL / "panel.json").read_bytes())["public_roots"]
    authority = json.loads(gzip.decompress((DATA / "compute/measurements.json.gz").read_bytes()))["roots"]
    cards = {(c["terrain"],c["expert_seat"]):c for c in PROTOCOL["games"] if c["split"] == "train"}
    for index,row in enumerate(schedule):
        seed,seat,step = row["seed"],row["seat"],row["t"]
        card = cards[row["terrain"],seat]
        assert card["seed"] == seed and row["receipt_sha256"]==card["sha256"]
        orig = next(r for r in authority if (r["terrain"],r["seat"],r["t"])==(row["terrain"],seat,step))
        yield index,card,orig,seat,step


def fixed_compare(mods, nets, order, obs, cfg, tracker, previous, seed, costs, preset=None):
    # References outlive the call; two budget decisions are the only things overridden.
    history = {}
    work = {"menu":[],"cut":[]} if preset is None else preset
    forward_calls=[]
    def run(label,capture=False):
        mod=mods[label]
        counts={"menu":0,"cut":0}
        seen=[]
        def gate(kind,value):
            if capture and preset is None:work[kind].append(value)
            ix=counts[kind];counts[kind]+=1
            return work[kind][ix]
        def forward(x,sid,scalar=None):
            result=nets[label](x,sid,scalar)
            seen.append((x,scalar,*result))
            return result
        mod._budget_cut=lambda value:gate("cut",value)
        mod._budget_menu=lambda value:gate("menu",value)
        mod._timed_forward=forward
        r=root_call(mod,tracker,previous,obs,cfg,seed,costs)
        assert {k:len(v) for k,v in work.items()}==counts
        if capture:forward_calls.extend(seen)
        assert len(seen)==len(forward_calls) and all(equal_tensors(a,b) for a,b in zip(forward_calls,seen))
        return r
    ref=run("parent",True)
    keys=["act0","opponent_samples","issued","menu","depth_by_arm","leaf_scores","best","candidates","depth","rollouts"]
    for arm in order:
        r=run(arm)
        assert all(ref[k]==r[k] for k in keys)
        history.setdefault(arm,[]).append({k:r[k] for k in ["wall_s","cpu_s"]})
    digest=hashlib.sha256()
    for call in forward_calls:
        for x in call:
            if x is not None:digest.update(x.tobytes())
    return dict(times=history,bitwise_equal=True,work=work,inputs_outputs_sha256=digest.hexdigest(),
                features=[(len(call[0]),call[1] is not None) for call in forward_calls],reference={k:ref[k] for k in keys})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--timing-repeat", action="store_true")
    repeat=parser.parse_args().timing_repeat
    previous_study=None
    if repeat:
        with gzip.open(HERE/"measurements.json.gz") as fid:previous_study=json.load(fid)
        assert len(previous_study["rows"])==12
    assert list(os.sched_getaffinity(0))==[4]
    assert all(os.environ.get(k)=="1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"))
    torch.set_num_interop_threads(1)
    raw={"parent":PARENT.read_bytes(),"candidate":build()}
    assert (HERE/"candidate.py").read_bytes()==raw["candidate"]
    mods={};nets={};free={}
    for arm,source in raw.items():
        mods[arm]=module(insert_hooks(source))
        torch.set_num_threads(1)
        nets[arm]=mods[arm]._net()
        assert type(nets[arm]).__name__=="_WinoNet" and mods[arm]._RNET is nets[arm]
        if not repeat:
            free[arm]=module(source)
            torch.set_num_threads(1)
            assert type(free[arm]._net()).__name__=="_WinoNet" and free[arm]._S_ABORT==.58
    rows=[]
    for index, card, state, seat, step in roots():
        m=mods["parent"]
        tr=m.BeliefTracker(m._Bel,seat)
        saved=[];prev=None
        for t,obs,cfg,order in free_entry.public_history(card):
            if step-3<=t<=step:
                saved.append((t,obs,{k:v for k,v in cfg.items() if k!="randomSeed"},copy.deepcopy(tr),prev))
            if t==step:break
            tr.step(obs["players"],t,obs["carbon"])
            tr.set_actions(obs["players"],order)
            prev=obs["players"]
        assert len(saved)==4
        schedule=["parent","candidate","candidate","parent"] if index%2==0 else ["candidate","parent","parent","candidate"]
        seed,costs=state["policy_rng"],state["starting_costs"]
        _,obs,cfg,tracker,previous=saved[-1]
        if repeat:
            first=previous_study["rows"][index]
            assert (card["seed"],seat,step)==(first["seed"],first["seat"],first["t"])
            schedule=(["parent","candidate"] if index%2==0 else ["candidate","parent"])*5
            fixed=fixed_compare(mods,nets,schedule,obs,cfg,tracker,previous,seed,costs,first["fixed"]["work"])
            assert fixed["inputs_outputs_sha256"]==first["fixed"]["inputs_outputs_sha256"]
            rows.append(dict(seed=card["seed"],seat=seat,t=step,pilot=first["pilot"],fixed=fixed))
            print(index,step,[r["wall_s"] for r in fixed["times"]["candidate"]],flush=True)
            continue
        fixed=fixed_compare(mods,nets,schedule,obs,cfg,tracker,previous,seed,costs)
        times={arm:[] for arm in raw}
        for arm in schedule:
            own_cost=costs;curve=[]
            for t,ob,conf,tr,hist in saved:
                r=root_call(free[arm],tr,hist,ob,conf,seed+step-t,own_cost)
                own_cost=r["ending_costs"]
                curve.append({k:r[k] for k in ["candidates","depth","rollouts","issued","depth_by_arm","best","wall_s","cpu_s","ending_costs"]})
            times[arm].append(curve)
        result=dict(seed=card["seed"],seat=seat,t=step,terrain=card["terrain"],pilot=index in (0,11),source_sha256=card["sha256"],root_rng=seed,prior_costs=costs,order=schedule,fixed=fixed,adaptive=times)
        rows.append(result)
        print(index,card["seed"],seat,step,{a:[(r[-1]["candidates"],r[-1]["depth"]) for r in v] for a,v in times.items()},flush=True)
        report=dict(source_sha256={k:sha(v) for k,v in raw.items()},protocol_sha256=sha((HERE/"protocol.json").read_bytes()),torch=torch.__version__,threads=torch.get_num_threads(),affinity=list(os.sched_getaffinity(0)),cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),rows=rows)
        (HERE/"measurements.json.gz").write_bytes(gzip.compress((json.dumps(report,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
    if repeat:
        report=dict(first_study_sha256=sha((HERE/"measurements.json.gz").read_bytes()),protocol_sha256=sha((HERE/"repeat_protocol.json").read_bytes()),rows=rows)
        (HERE/"repeat.json.gz").write_bytes(gzip.compress((json.dumps(report,separators=(",",":"),allow_nan=False)+'\n').encode(),mtime=0))


if __name__=="__main__":
    main()
