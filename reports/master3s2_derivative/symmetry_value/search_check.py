"""Run the prospectively registered finalist-only 180-degree critic test against its own parent."""
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
DATA=HERE.parent/"opponent_policy"
sys.path[:0]=[str(DATA),str(DATA/"search_probe")]
from full_search_check import root_call
from build import build,sha

PROTOCOL_SHA="1a2d96babdbacccf41724a830c5fd3563801559d73793cc0a5fb4cdf80834cc9"
CANDIDATE_SHA="69b6730dcc990404bae939b5978126c580a4731b4c0559b7e6c499f387e7597b"


def normalize(row,trace):
    # The native _S_REC reports a per-menu vector, including nonfinal candidates at shallower depths.
    # Replacing the final candidates with their recorded pre-evaluation scores recovers the original comparator.
    scores=list(row["leaf_scores"])
    for i,val in zip(trace.get("finalists",[]),trace.get("before",[]),strict=True):
        assert scores and row["depth_by_arm"][i]==8
        scores[i]=val
    row["before_scores"]=scores
    if scores:
        relevant=[i for i,h in enumerate(row["depth_by_arm"]) if h==row["depth"]]
        # A stopped-at-prune native call may retain only a subset of equal-depth rows.
        # Only H=8 rows are ever re-evaluated, at which point the actual alive set is recorded above.
        if trace.get("finalists"):
            relevant=trace["finalists"]
            bi=max(relevant,key=lambda i:scores[i])
            pre=bi if scores[bi]>scores[0] else 0
            row["before_order"]=row["menu"][pre]
        else:
            row["before_order"]=row["issued"]
    else:row["before_order"]=row["issued"]
    row["rescore"]=trace
    return row


def assess(rows):
    a,b,x,y=[rows[i] for i in ("P1","P2","C1","C2")]
    same=all(all(r[k]==a[k] for r in (b,x,y)) for k in
             ("menu","depth_by_arm","before_scores","depth","rollouts","act0","opponent_samples","before_order"))
    d=a["issued"]!=x["issued"]
    commits=all(r["rescore"].get("stage")=="committed" for r in (x,y))
    stable=a["issued"]==b["issued"] and x["issued"]==y["issued"]
    change=bool(same and d and commits and stable and all(r["issued"]!=r["before_order"] for r in (x,y)))
    return dict(exact_first_path=same,physical_changed=d,repeated_final_orders=stable,
                both_committed=commits,attributable_changed=change)


def main():
    os.sched_setaffinity(0,{4})
    for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
    os.environ["CARBON_DEBUG"]="1"
    import torch
    torch.set_num_interop_threads(1)
    from original_policy import PROTOCOL,module,free_entry
    pb=(HERE/"search_protocol.json").read_bytes()
    assert sha(pb)==PROTOCOL_SHA
    plan=json.loads(pb)
    parent=(HERE.parent/"cold_start/candidate.py").read_bytes()
    candidate=(HERE/"candidate.py").read_bytes()
    assert sha(parent)==plan["base"] and sha(candidate)==CANDIDATE_SHA and build()==candidate
    assert sha((DATA/"protocol.json").read_bytes())==plan["sources_sha256"]
    mods={k:module(raw) for k,raw in (("P",parent),("C",candidate))}
    torch.set_num_threads(1)
    assert all(m._TEMP==1 and m._SEED is None and m._SEARCH_OK and m._RNET is m._NET and m._S_ABORT==.58 for m in mods.values())
    base=mods["P"]
    init=list(base._S_COST)
    indexed={r["path"]:{} for r in plan["roots"]}
    for r in plan["roots"]:indexed[r["path"]][r["step"]]=r
    by_path={c["path"]:c for c in PROTOCOL["games"] if c["split"]=="train"}
    allrows=[]
    for path,targets in indexed.items():
        card=by_path[path]; seat=card["expert_seat"]
        assert sha((HERE.parents[2]/path).read_bytes())==card["sha256"]
        tr=base.BeliefTracker(base._Bel,seat)
        prev=None
        for t,obs,cfg,archived in free_entry.public_history(card):
            upcoming=next((k for k in targets if k-3<=t<=k),None)
            if upcoming is not None:
                ticket=targets[upcoming];cfg={k:v for k,v in cfg.items() if k!="randomSeed"}
                if t==upcoming-3:cost=init
                if t!=upcoming:
                    h=hashlib.sha256(f'symmetry-warm/{ticket["terrain"]}/{seat}/{t}'.encode()).digest()
                    cost=root_call(base,tr,prev,obs,cfg,int.from_bytes(h[:4],"big"),cost)["ending_costs"]
                else:
                    arms={}
                    for key in ticket["order"]:
                        m=mods[key[0]]
                        if key.startswith("C"):m._SYM_DIAGNOSTIC={}
                        result=root_call(m,tr,prev,obs,cfg,ticket["policy_rng"],cost)
                        trace=copy.deepcopy(m._SYM_DIAGNOSTIC) if key.startswith("C") else {}
                        if key.startswith("C"):
                            m._SYM_DIAGNOSTIC=None
                        arms[key]=normalize(result,trace)
                    tested=assess(arms)
                    allrows.append(dict(**ticket,arms=arms,assessment=tested,starting_cost=cost))
                    print(len(allrows),ticket["seed"],seat,t,arms["C1"]["rescore"].get("stage","primary_only"),tested["attributable_changed"],flush=True)
            tr.step(obs["players"],t,obs["carbon"])
            tr.set_actions(obs["players"],archived);prev=obs["players"]
    assert len(allrows)==len(plan["roots"])==32
    phases=Counter();admitted=[]
    for row in allrows:
        for k in ("C1","C2"):
            arm=row["arms"][k]
            marker=arm["rescore"]
            phases[marker.get("stage","primary_only")]+=1
            if marker.get("evaluated",0):admitted.append(arm["wall_s"])
    late=sum(t>.58 for t in admitted)
    counts={k:sum(r["assessment"][k] for r in allrows) for k in allrows[0]["assessment"]}
    report=dict(protocol_sha256=PROTOCOL_SHA,candidate=CANDIDATE_SHA,parent=plan["base"],verifier=sha(Path(__file__).read_bytes()),
                cpu_affinity=list(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),torch_version=torch.__version__,host=platform.platform(),
                cgroup_cpu=Path("/sys/fs/cgroup/cpu.max").read_text().strip(), measured_utc=time.strftime("%FT%TZ",time.gmtime()),
                roots=allrows,counts=counts,stages=phases,admitted_calls=len(admitted),admitted_over_058=late,
                max_admitted_public_s=max(admitted,default=None),root_mechanism_pass=counts["attributable_changed"]>0 and not late)
    with (HERE/"search_roots.json.gz").open("xb") as f:
        f.write(gzip.compress((json.dumps(report,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode(),mtime=0))
    print(counts,phases,late,report["root_mechanism_pass"],flush=True)


if __name__=="__main__":
    main()
