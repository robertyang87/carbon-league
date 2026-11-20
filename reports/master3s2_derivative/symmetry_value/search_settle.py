"""Evaluate all registered calls, plus near ties and one-step physical equivalence of root orders."""
from collections import Counter
import gzip
import json
from pathlib import Path
import random
from statistics import median
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/"opponent_policy"
sys.path.insert(0,str(DATA))
from original_policy import free_entry,PROTOCOL


def delta(a):
    tr=a["rescore"]
    if tr.get("stage")!="committed" or a["before_order"]==a["issued"]:return None
    old=a["menu"].index(a["before_order"])
    new=a["menu"].index(a["issued"])
    i=tr["finalists"].index(old);j=tr["finalists"].index(new)
    return [tr[k][j]-tr[k][i] for k in ("before","after")]


def physical(row, actions):
    # Independent official one-step replay; same simultaneous opposing order in both arms.
    # Check both the simulated shared response and the archived real opposing order.
    raw=(HERE.parents[2]/row["path"]).read_bytes()
    tape=json.loads(gzip.decompress(raw))
    step=row["step"];seat=row["seat"]
    orders=[row["arms"]["C1"]["opponent_samples"][0],tape["actions"][step][1-seat]]
    results=[]
    for o in orders:
        states=[]
        for action in actions:
            cfg=dict(tape["game_configuration"],randomSeed=row["seed"])
            random.seed(row["seed"]);np.random.seed(row["seed"])
            env=free_entry.official_rules(HERE.parents[2])("carbon",configuration=cfg)
            for joint in tape["actions"][:step]:env.step(joint,[])
            joint=[None,None];joint[seat]=action;joint[1-seat]=o
            env.step(joint,[])
            state=env.states[0].observation
            states.append(free_entry.digest([state.step,state.full_carbon,state.players,state.trees,
                                             [(a.reward,a.status) for a in env.states]]))
        results.append(dict(same_official_next=states[0]==states[1],hashes=states))
    return results


def main():
    report=json.loads(gzip.decompress((HERE/"search_roots.json.gz").read_bytes()))
    rows=[]
    stage=Counter()
    hooks=[]
    for r in report["roots"]:
        change=r["assessment"]["attributable_changed"]
        ds=[delta(r["arms"][k]) for k in ("C1","C2")]
        traces=[r["arms"][k]["rescore"] for k in ("C1","C2")]
        status=[tr.get("stage","primary_complete_without_hook") for tr in traces]
        for k in status:stage[k]+=1
        checks=physical(r,(r["arms"]["C1"]["before_order"],r["arms"]["C1"]["issued"])) if ds[0] else None
        rr=dict(seed=r["seed"],seat=r["seat"],step=r["step"],status=status,**r["assessment"],
                both_repeat_advantage=ds,all_original_and_new_abs_gt_1e_5=bool(change and all(d and min(map(abs,d))>1.e-5 for d in ds)),
                first_trial_physics=checks)
        for side in ("C1","C2"):
            a=r["arms"][side]
            if a["rescore"].get("evaluated"):hooks.append(a["wall_s"])
        rows.append(rr)
    summary=dict(counts=report["counts"],stages=stage,admitted_calls=len(hooks),
                 admitted_public_p95_s=float(np.percentile(hooks,95)) if hooks else None,
                 admitted_public_p99_s=float(np.percentile(hooks,99)) if hooks else None,
                 admitted_public_max_s=max(hooks,default=None),admitted_over_058=report["admitted_over_058"],
                 repeat_adjustments_gt_1e_5=sum(r["all_original_and_new_abs_gt_1e_5"] for r in rows),
                 mother_controls_equal=sum(r["arms"]["P1"]["issued"]==r["arms"]["P2"]["issued"] for r in report["roots"]),
                 roots=rows,protocol_gate=report["root_mechanism_pass"])
    print(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
