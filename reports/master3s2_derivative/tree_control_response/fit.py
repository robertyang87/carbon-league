"""Fit expected first-ownership cash and tenure, grouping by the old, signed physical-map split."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"ladder/tree_payback"
def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix==".gz" else data)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def target(rows):
    return np.array([[r["net"],r["end_t"]-r["t"]] for r in rows],dtype=float)
def main():
    a,b=[[r for r in read(OLD/(split+"_trees.json.gz")) if r["usable"]] for split in ("train","held")]
    train_maps=set(r["terrain"] for r in a);held_maps=set(r["terrain"] for r in b)
    registry=HERE.parent/"ladder/exposed_ladder_maps.json"
    exposures={r["opening_carbon_bytes_sha256"] for r in read(registry)}
    if len(train_maps)!=24 or len(held_maps)!=8 or train_maps & held_maps or (train_maps|held_maps)&exposures:raise ValueError("Terrain collision")
    x=np.array([r["X"] for r in a]);y=target(a)
    maps=np.array([r["terrain"] for r in a])
    m=np.array([1./(len(train_maps)*(maps==k).sum()) for k in maps])
    center=m@x;scale=np.maximum(.05,np.sqrt(m@(x-center)**2));mean=m@y
    z=(x-center)/scale
    w=np.linalg.solve(z.T@(m[:,None]*z)+0.1*np.eye(x.shape[1]),z.T@(m[:,None]*(y-mean)))
    score=[]
    for h in sorted(held_maps):
        rs=[r for r in b if r["terrain"]==h];yx=target(rs)
        p=mean+((np.array([r["X"] for r in rs])-center)/scale)@w
        score.append(dict(terrain=h,n=len(rs),mae_mean=np.abs(yx-mean).mean(axis=0).tolist(),mae_pred=np.abs(yx-p).mean(axis=0).tolist()))
    doc=dict(method="two independent first-tenure expected cash and holding steps, map-equal ridge λ=0.1 with 23 public preplant coordinates; no win labels",
             train_sha=sha(OLD/"train_trees.json.gz"),held_sha=sha(OLD/"held_trees.json.gz"),frozen_classifier_sha=sha(OLD/"weights.json"),
             fit_code=sha(Path(__file__)),exposure_source=sha(registry),excluded_exposed=len(exposures),train_maps=sorted(train_maps),held_maps=sorted(held_maps),
             train_n=len(a),held_n=len(b),feature_mean=center.tolist(),feature_scale=scale.tolist(),means=mean.tolist(),coef=w.tolist(),held_results=score)
    dest=HERE/"weights.json";dest.write_text(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n")
if __name__=="__main__":main()
