"""Three paths over the exact captured network workload, balanced in order; no new games or training."""
from collections import Counter
import itertools
import json
from pathlib import Path
import platform
from statistics import median
import time
from capture import HERE,setup,sha


def timing(run):
    a,b=time.perf_counter_ns(),time.process_time_ns()
    run()
    return ((time.perf_counter_ns()-a)/1.e9,(time.process_time_ns()-b)/1.e9)


def main():
    plan,m=setup()
    import numpy as np
    import torch
    from forward import evaluate,frozen_targets
    records=json.loads((HERE/"panel.json").read_bytes())
    assert sha((HERE/"panel.npz").read_bytes())==records["features_sha256"]
    assert sha((HERE/"protocol.json").read_bytes())==records["protocol_sha256"]
    arms=plan["arms"]
    permutations=list(itertools.permutations(arms))
    assert len(permutations)==plan["repetitions"]
    net=m._NET
    maxerr=Counter()
    ranks=Counter()
    totals=Counter()
    cases=[]
    with np.load(HERE/"panel.npz",allow_pickle=False) as file:
        for ix,row in enumerate(records["public_roots"]):
            inputs=[]
            for item in row["batches"]:
                key=item["key"]
                x=file[key].astype(np.float32)*np.float32(.01)
                sc=file[key+"s"] if item["scalar"] else None
                item_args=(net,x,item["sid"],sc,item["kind"])
                inputs.append(item_args)
                all_outputs={arm:evaluate(*item_args,arm,net.hw,net.hb) for arm in arms}
                for left,right,names in (("original","shared",("own_worker","own_base","own_value",None,None)),
                                         ("separate","shared",("split_own_worker","split_own_base","split_own_value","split_opp_worker","split_opp_base"))):
                    for label,u,v in zip(names,all_outputs[left],all_outputs[right]):
                        if label is None or u is None:continue
                        scales=np.max(np.abs(u))
                        err=float(np.max(np.abs(u-v)))
                        maxerr[label]=max(maxerr[label],err)
                        maxerr[label+"_rel_global"]=max(maxerr[label+"_rel_global"],err/(1+float(scales)))
                        if label.endswith(("base","worker")):
                            ranks[label+"_argmax_changed"]+=int(np.count_nonzero(u.argmax(-1)!=v.argmax(-1)))
                            ranks[label+"_positions"]+=u.shape[0]*u.shape[1]
                if item["kind"] in ("joint","other"):
                    opp=x[1::2] if item["kind"]=="joint" else x
                    shallow,lw,lb=frozen_targets(net,np.ascontiguousarray(opp),item["sid"])
                    assert shallow.shape==(len(opp),225,96)
                    for name,target,expected in zip(("worker","base"),(lw,lb),all_outputs["original"][3:]):
                        maxerr["training_target_"+name]=max(maxerr["training_target_"+name],float(np.max(np.abs(target-expected))))
            def sweep(arm):
                for args in inputs:evaluate(*args,arm,net.hw,net.hb)
            for arm in arms:sweep(arm)
            times={arm:[] for arm in arms}
            for j in range(len(permutations)):
                for arm in permutations[(ix+j)%len(permutations)]:
                    times[arm].append(timing(lambda:sweep(arm)))
            per={a:median(t[0] for t in ts) for a,ts in times.items()}
            actual_net=sum(b["original_live_forward_ns"] for b in row["batches"])/1.e9
            residual=row["public_wall"]-actual_net
            assert residual>=0
            for arm in arms:
                totals[arm]+=per[arm]
                totals[arm+"_cpu"]+=median(t[1] for t in times[arm])
                totals[arm+"_fixed_public_est"]+=per[arm]+residual
            cases.append(dict(seed=row["seed"],seat=row["seat"],step=row["t"],times=times,medians=per,source_residual_s=residual))
            print(row["seed"],row["seat"],row["t"],[round(per[a],3) for a in arms],flush=True)
    report=dict(panel_sha256=sha((HERE/"panel.json").read_bytes()),forward_sha256=sha((HERE/"forward.py").read_bytes()),timing_sha256=sha(Path(__file__).read_bytes()),
                platform=platform.platform(),torch=torch.__version__,threads=torch.get_num_threads(),affinity=plan["affinity"],
                absolute_errors=maxerr,board_argmax=ranks,sum_root_medians=totals,rows=cases)
    with (HERE/"measurements.json").open("x") as f:
        json.dump(report,f,sort_keys=True,allow_nan=False,indent=2)
        f.write("\n")
    print(dict(totals),dict(maxerr),flush=True)


if __name__=="__main__":
    main()
