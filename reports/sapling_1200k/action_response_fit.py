"""Fit a map-weighted ridge action difference, freeze it, and audit a one-shot map-level test."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import action_response as ar

OUT=ar.OUT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(split):
    cards=json.loads((OUT/"plan.json").read_bytes())["maps"]
    selected=[]
    for c in cards:
        if c["split"]!=split:continue
        for task in c["tasks"]:
            path=ar.resultfile(c,task)
            row=json.loads(gzip.decompress(path.read_bytes()))
            if row["card"]!=c or row["task"]!=task or row["plan_sha256"]!=digest(OUT/"plan.json"):
                raise ValueError("source row differs from plan")
            if "root" not in row:continue
            if any(row[arm]["terminal_status"]!=["DONE","DONE"] for arm in ("parent","alternative")):
                raise ValueError("incomplete scored action label")
            selected.append(dict(id=c["id"],seed=c["seed"],family=task["family"],seat=task["seat"],
                                 step=row["root"]["step"],orientation=row["root"]["direction"],
                                 static=row["root"]["tree_static_alt"],x=row["root"]["feature"],
                                 du=row["delta_u"],dm=row["delta_margin"],raw_sha256=digest(path)))
    return selected


def metric(r,data):
    groups=defaultdict(list)
    for x,p in zip(r,data):
        groups[x["id"]].append((float(p),x["du"],x["dm"]))
    if not groups:return {"maps":0,"roots":0,"changed_maps":0,"mean_delta_u":None,"mean_delta_margin":None}
    def means(i):
        return np.array([np.mean([p*vals[i] for p,*vals in group]) for group in groups.values()])
    u,margin=means(0),means(1)
    omit=(u.sum()-u)/(len(u)-1) if len(u)>1 else None
    return {"maps":len(groups),"roots":len(r),"changed_maps":sum(any(p>0 for p,_,_ in v) for v in groups.values()),
            "expected_changes":float(sum(data)),"mean_delta_u":float(u.mean()),"mean_delta_margin":float(margin.mean()),
            "positive_zero_negative_maps":[int((u>0).sum()),int((u==0).sum()),int((u<0).sum())],
            "leave_one_map_out_mean_u_range":[float(omit.min()),float(omit.max())] if omit is not None else None}


def scores(model,r):
    if not r:return np.array([],dtype=float)
    x=np.asarray([p["x"] for p in r],float)
    return (x/np.asarray(model["scale"]))@model["coefficients"]


def fit():
    model_path=OUT/"model.json";freeze_path=OUT/"freeze.json"
    if freeze_path.exists():
        lock=json.loads(freeze_path.read_bytes())
        if digest(model_path)!=lock["model_sha256"] or digest(OUT/"plan.json")!=lock["plan_sha256"]:
            raise ValueError("existing freeze has wrong identities")
        return
    p=json.loads((OUT/"plan.json").read_bytes())
    if any(ar.resultfile(c,t).exists() for c in p["maps"] if c["split"]=="test" for t in c["tasks"]):
        raise ValueError("test labels present before fitting")
    tr,va=rows("train"),rows("val")
    n=Counter(q["id"] for q in tr)
    if not n:raise ValueError("no eligible training examples")
    x=np.array([r["x"] for r in tr],float);y=np.array([r["du"] for r in tr],float)
    w=np.array([1./n[r["id"]] for r in tr])
    scale=np.maximum(.25,np.sqrt(np.average(x*x,weights=w,axis=0)))
    a=x/scale
    coef=np.linalg.solve(a.T@(w[:,None]*a)+10.*np.eye(a.shape[1]),a.T@(w*y))
    model={"features":p["features"],"scale":scale.tolist(),"coefficients":coef.tolist(),"intercept":0.,
           "train_n":len(tr),"train_maps":len(n),"ridge":10.,"plan_sha256":digest(OUT/"plan.json")}
    if not np.all(np.isfinite(coef)):raise ValueError("nonfinite fitted weights")
    val=[]
    for threshold in (0.,.02,.05):
        val.append({"threshold":threshold,**metric(va,scores(model,va)>threshold)})
    ok=[v for v in val if v["changed_maps"]>=2 and v["mean_delta_u"]>0 and v["mean_delta_margin"]>=0]
    model["threshold"]=max(ok,key=lambda v:(v["mean_delta_u"],v["threshold"]))["threshold"] if ok else None
    model["validation"]=val
    ar.b.write(model_path,model)
    ar.b.write(freeze_path,{"model_sha256":digest(model_path),"plan_sha256":digest(OUT/"plan.json"),
                            "train_val_tapes":{r["id"]+"/"+r["family"]:r["raw_sha256"] for r in tr+va},
                            "code_sha256":{p.name:digest(p) for p in (Path(__file__),Path(ar.__file__),Path(ar.b.__file__))}})
    print("frozen",digest(model_path),model["threshold"],flush=True)


def evaluate():
    model=json.loads((OUT/"model.json").read_bytes())
    frozen=json.loads((OUT/"freeze.json").read_bytes())
    if digest(OUT/"model.json")!=frozen["model_sha256"]:raise ValueError("model bytes differ")
    res={}
    for split in ("train","val","test"):
        r=rows(split);pred=scores(model,r);n=len(r)
        true=np.array([q["du"] for q in r],float)
        map_roots=Counter(q["id"] for q in r)
        weights=np.array([1./map_roots[q["id"]] for q in r],float)
        diagnostics={} if not n else {
            "map_weighted_mae":float(np.average(abs(pred-true),weights=weights)),
            "zero_prediction_mae":float(np.average(abs(true),weights=weights)),
            "strict_positive_label_roots":int((true>0).sum()),
            "strict_positive_prediction_roots":int((pred>0).sum()),
            "map_weighted_direction_accuracy":float(np.average((pred>0)==(true>0),weights=weights))}
        methods={"retain_parent":np.zeros(n),"random_50_expected":np.full(n,.5),
                 "static_tree_only":np.array([q["static"] for q in r],bool),"raw_positive_head":pred>0,
                 "validation_frozen":pred>model["threshold"] if model["threshold"] is not None else np.zeros(n)}
        by_family={}
        for fam in ar.FAMILIES:
            ids=[i for i,q in enumerate(r) if q["family"]==fam]
            by_family[fam]={name:metric([r[i] for i in ids],v[ids]) for name,v in methods.items()}
        res[split]={"overall":{name:metric(r,v) for name,v in methods.items()},"by_family":by_family,
                    "training_head_prediction":diagnostics,
                    "rows":[{k:v for k,v in q.items() if k!="x"}|{"public_action_difference":q["x"],"pred_delta_u":float(z)}
                            for q,z in zip(r,pred)]}
    ar.b.write(OUT/"evaluation.json",{"model_sha256":frozen["model_sha256"],"sets":res})
    print(res["test"]["overall"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("stage",choices=("fit","evaluate"))
    if ap.parse_args().stage=="fit":fit()
    else:evaluate()
