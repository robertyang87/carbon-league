"""Compare stock calls with timing probes; exact equality is measured after the call returns."""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/"search_probe"))
from full_search_check import root_call

STEPS=(20,85,215)
TARGET_MAPS=2
def hashb(b):
    return hashlib.sha256(b).hexdigest()
def get(path):
    return Path(path).read_text().strip()

class Probes:
    def __init__(self,m):
        self.saved=[]
        self.ns=Counter()
        self.rows=[]
        self.searching=False
        self.model=m
    def patch(self,owner,name,f):
        self.saved.append((owner,name,getattr(owner,name)));setattr(owner,name,f)
    def __enter__(self):
        m=self.model
        def timed(fn,label):
            def capture(*args,**kwargs):
                start=time.perf_counter_ns()
                result=fn(*args,**kwargs)
                self.ns[label]+=time.perf_counter_ns()-start
                return result
            return capture
        for owner,name,label in [(m,"_turn","simulation"),(m,"encode","encoding"),(m,"_sample","sampling"),
                                 (m._Policy,"forward","own_root_neural")]:
            self.patch(owner,name,timed(getattr(owner,name),label))
        klass=type(m._NET);native=klass.__call__
        def call(instance,x,sid,sc=None):
            start=time.perf_counter_ns();result=native(instance,x,sid,sc);elapsed=time.perf_counter_ns()-start
            # These argument arrays are allocated fresh by look/other and kept until after the wall-clock search.
            self.rows.append((x,sid,sc,elapsed))
            self.ns["search_neural"]+=elapsed
            return result
        self.patch(klass,"__call__",call)
        return self
    def __exit__(self,*args):
        for owner,name,fn in reversed(self.saved):setattr(owner,name,fn)
    def report(self):
        seen=set();full=set();counts=Counter();batches=[]
        simulated_reuse_ns=0.;touched_batch_ns=0.
        for x,sid,scalar,elapsed in self.rows:
            earlier=set(seen); local=set();same=0;batch_full=0
            for row,vector in enumerate(x):
                key=(int(sid),vector.dtype.str,tuple(vector.shape),hashb(vector.tobytes()))
                duplicate=key in seen
                counts["rows"]+=1;counts["exact_trunk_repeats"]+=duplicate
                counts["seen_in_earlier_pass"]+=int(key in earlier)
                counts["seen_within_this_pass"]+=int(key in local)
                scalar_key=None if scalar is None else hashb(scalar[row].tobytes())
                complete=(key,scalar_key) in full or (scalar is None and duplicate)
                counts["full_requested_outputs_reusable"]+=int(complete)
                batch_full+=int(complete);same+=duplicate
                seen.add(key);full.add((key,scalar_key));local.add(key)
            counts["calls"]+=1
            simulated_reuse_ns+=elapsed*same/len(x)
            touched_batch_ns+=elapsed if same else 0
            batches.append(dict(rows=len(x),duplicate_trunks=same,full_outputs=batch_full,wall_ms=elapsed/1.e6,critic=scalar is not None))
        return dict(counts=counts,stage_ns=self.ns,batches=batches,linear_trunk_saving_ns=simulated_reuse_ns,
                    whole_affected_batch_ns=touched_batch_ns)

def main(cpu):
    os.sched_setaffinity(0,{cpu})
    for e in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[e]="1"
    os.environ["CARBON_DEBUG"]="1"
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from original_policy import original,module,free_entry,PROTOCOL
    native=module(original())
    assert native._S_ABORT==.58 and native._RNET is native._NET
    games=[r for r in PROTOCOL["games"] if r["split"]=="train"]
    maps=sorted({r["terrain"] for r in games})[:TARGET_MAPS]
    cards=sorted((r for r in games if r["terrain"] in maps),key=lambda r:(r["terrain"],r["expert_seat"]))
    rows=[]
    warm_cost=list(native._S_COST)
    for card in cards:
        me=card["expert_seat"]
        tracker=native.BeliefTracker(native._Bel,me)
        prev=None
        for t,obs,cfg,issued in free_entry.public_history(card):
            active=next((s for s in STEPS if s-3<=t<=s),None)
            if active is not None:
                cfg={k:v for k,v in cfg.items() if k!="randomSeed"}
                seed=int.from_bytes(hashlib.sha256(f'original-compute-v1/{card["terrain"]}/{me}/{t}'.encode()).digest()[:4],"big")
                if t==active-3:cost=warm_cost
                if t<active:
                    trace=root_call(native,tracker,prev,obs,cfg,seed,cost)
                    cost=trace["ending_costs"]
                else:
                    order=("probe","control") if len(rows)%2==0 else ("control","probe")
                    results={}
                    for name in order:
                        if name=="probe":
                            with Probes(native) as probes:
                                results[name]=root_call(native,tracker,prev,obs,cfg,seed,cost)
                            stats=probes.report()
                        else:
                            results[name]=root_call(native,tracker,prev,obs,cfg,seed,cost)
                    def summary(r):
                        return {k:r[k] for k in ("candidates","depth","rollouts","native_aborted","wall_s","cpu_s","single")}
                    a,b=results["probe"],results["control"]
                    fields=("menu","sorted_depths","depth","rollouts","native_aborted","issued")
                    rows.append(dict(terrain=card["terrain"],seed=card["seed"],seat=me,t=t,policy_rng=seed,
                        source_commands_sha=card["actions_sha256"],starting_costs=cost,order=order,
                        probe=summary(a),control=summary(b),same_work_and_action=all(a[k]==b[k] for k in fields),measure=stats))
            tracker.step(obs["players"],t,obs["carbon"]);tracker.set_actions(obs["players"],issued);prev=obs["players"]
        print(card["seed"],me,flush=True)
    counts=Counter();times=Counter()
    for r in rows:
        counts.update(r["measure"]["counts"]);times.update(r["measure"]["stage_ns"])
    total=sum(r["probe"]["wall_s"] for r in rows)
    summary=dict(root_calls=len(rows),opening_histories=4,train_maps=maps,measurement_scope="0.58 s native unmodified public agent. Hashes post-return, no network cache, within-root only; three preceding real-wall public calls remeasure this host's latest estimator. Individual components measured at callable boundaries.",
        rows=counts,stage_wall_s={k:v/1e9 for k,v in times.items()},profile_total_wall_s=total,
        same_work_and_action=sum(r["same_work_and_action"] for r in rows),linear_upper_saving_s=sum(r["measure"]["linear_trunk_saving_ns"] for r in rows)/1e9,
        full_affected_batch_ceiling_s=sum(r["measure"]["whole_affected_batch_ns"] for r in rows)/1e9)
    report=dict(source=hashb(original()),source_protocol=hashb((HERE.parent/"protocol.json").read_bytes()),profile_sha256=hashb(Path(__file__).read_bytes()),
                measured_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),steps=STEPS,host=platform.platform(),
                group_cpu=get("/sys/fs/cgroup/cpu.max"),group_memory=get("/sys/fs/cgroup/memory.max"),affinity=list(os.sched_getaffinity(0)),
                torch_version=torch.__version__,configured_torch_threads=torch.get_num_threads(),search_net=type(native._NET).__name__,summary=summary,roots=rows)
    dst=HERE/"measurements.json.gz"
    if dst.exists():raise ValueError("Existing run cannot be overwritten")
    dst.write_bytes(gzip.compress((json.dumps(report,indent=2,allow_nan=False,sort_keys=True)+"\n").encode(),mtime=0))
    print(summary,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--cpu",type=int,required=True);main(ap.parse_args().cpu)
