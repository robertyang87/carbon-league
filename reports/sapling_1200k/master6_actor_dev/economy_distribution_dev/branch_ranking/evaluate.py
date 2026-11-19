"""Compare paired response ranks and regrets with 0-delta and immediate-cash controls."""
import json
from pathlib import Path
import hashlib
from collections import defaultdict
import numpy as np

HERE=Path(__file__).resolve().parent
EPS=.01
CH=("own_delivery","own_tree","rival_delivery","rival_tree")


def paired_score(P,T,maps):
    P=np.asarray(P);T=np.asarray(T);maps=np.asarray(maps)
    # One cent is the official score resolution. Half-credit/half-regret for leaf ties.
    sign=np.where(abs(P)<=EPS,0,np.sign(P))
    true=np.where(abs(T)<=EPS,0,np.sign(T))
    active=true!=0
    hits=np.where(sign==0,.5,(sign==true).astype(float))
    regret=np.maximum(T,0)-np.where(sign==0,.5*T,np.where(sign>0,T,0))
    return_zero=np.maximum(T,0)
    def avg(x,idx):
        return float(np.mean([x[(maps==k)&idx].mean() for k in np.unique(maps[idx])])) if idx.any() else None
    every=np.ones(len(T),bool)
    return {"roots":len(T),"maps":len(set(maps)),"real_non_ties":int(active.sum()),
            "predict_non_ties":int((sign!=0).sum()),"hits_on_real_nonties":float(hits[active].sum()),
            "root_unweighted_nontie_accuracy":float(hits[active].mean()) if active.any() else None,
            "map_equal_nontie_accuracy":avg(hits,active),"map_equal_mae":avg(abs(P-T),every),
            "zero_delta_mae":avg(abs(T),every),"map_equal_regret":avg(regret,every),
            "constant_split_tie_regret":avg(abs(T)*.5,every),"always_parent_regret":avg(return_zero,every),
            "mean_predicted_change":avg(P,every),"mean_observed_change":avg(T,every),
            "map_equal_abs_predicted_change":avg(abs(P),every),
            "map_gain_over_constant":{str(k):float(np.mean(abs(T[maps==k])*.5-regret[maps==k])) for k in sorted(set(maps))}}


def deltas(row,scenario,model,h):
    forecast=row["forecast"]; obs=row["actual"];h=str(h)
    def calc(arm):
        f=forecast[arm][scenario];q=f["readouts"][model][h];a=obs[arm];win=a["windows"][h]
        preds={**dict(zip(CH,q["channels"])),"gross":q["net_gross_receipts"],"with_root":q["with_root_net_treasury"],
               "immediate_only":f["root_net_treasury"]}
        real={**dict(zip(CH,win["successor"]["channels"])),"gross":win["successor"]["net_gross_receipts"],
              "treasury":win["successor"]["net_treasury_change"],
              "whole_treasury":win["root_plus_successor"]["net_treasury_change"],
              "exact_root_h_treasury":win["from_root"]["net_treasury_change"],
              "official_margin":a["official_margin"]}
        return preds,real
    def diff(b,a):return {k:b[k]-a[k] for k in b}
    p1,t1=calc("alternative");p0,t0=calc("parent")
    return diff(p1,p0),diff(t1,t0)


def main():
    source=HERE/"results.json";d=json.loads(source.read_bytes());groups=defaultdict(list)
    detail=[]
    for row in d["rows"]:
        family="freight" if "early_lane" in row["source"] else "trees"
        x={"seed":row["seed"],"seat":row["seat"],"step":row["step"],"old":row["old"],"new":row["new"],"family":family,"comparisons":{}}
        for scenario in ("public_estimate","actual_successor"):
            for model in ("candidate","control"):
                for h in (10,30):
                    p,t=deltas(row,scenario,model,h)
                    x["comparisons"][f"{scenario}_{model}_{h}"]={"forecast":p,"actual":t}
                    pairs=[(c,c) for c in CH]+[("gross","gross"),("gross","treasury"),("with_root","whole_treasury"),
                           ("with_root","exact_root_h_treasury"),("with_root","official_margin"),
                           ("gross","official_margin"),("immediate_only","official_margin"),("immediate_only","whole_treasury")]
                    for group in (family,"all"):
                        for method,target in pairs:
                            groups[group,scenario,model,h,method,target].append((p[method],t[target],row["seed"]))
        detail.append(x)
    summaries={}
    for key,values in groups.items():
        summaries["/".join(map(str,key))]=paired_score(*zip(*values))
    results={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"cent_tie":EPS,
             "cases":detail,"comparisons":summaries,"excluded":d["excluded"]}
    (HERE/"rankings.json").write_text(json.dumps(results,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")


if __name__=="__main__":
    main()
