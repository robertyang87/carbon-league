"""Exploratory calibration by public rival workforce, holding the predicted model and data fixed."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
CACHE=Path("/workspace/sapling-all-top5-20261119/replays")


def main():
    sel=json.loads((HERE/"selection.json").read_bytes())
    source=HERE/"predictions.npz"
    with np.load(source) as z:
        idx=6
        use=z["valid_horizon_targets"][:,idx]
        gold=np.expm1(z["target_log1p"][:,idx])
        pred={arm:np.expm1(np.maximum(0,z[arm+"_income_standardized"][:,idx]*z["training_std"][idx]+z["training_mean"][idx])) for arm in ("control","candidate")}
        collector_count=np.zeros(len(gold),int)
        roots={}
        for g in sel["cards"]:
            packed=(CACHE/(g["id"]+".json.gz")).read_bytes()
            if hashlib.sha256(packed).hexdigest()!=g["sha256"]:
                raise ValueError("wrong replay")
            f=json.loads(gzip.decompress(packed))["steps"]
            for i in np.flatnonzero((z["gid"]==g["gid"]) & use):
                t=int(z["step"][i]);assets=f[t][0]["observation"]["players"][1-g["seat"]]
                workers={k:v for k,v in assets[2].items() if v[2]=="COLLECTOR"}
                collector_count[i]=len(workers)
                roots[int(i)]={"id":g["id"],"opponent":g["opponent"],"step":t,"rival_seat":1-g["seat"],
                               "rival_cash_before":assets[0],"rival_collectors_before":workers,
                               "rival_cargo_before":sum(w[1] for w in workers.values()),
                               "actual_30_steps":float(gold[i]),"control":float(pred["control"][i]),"candidate":float(pred["candidate"][i])}
        rows={}
        for label,mask in (("none",collector_count==0),("one_to_two",(collector_count>0)&(collector_count<=2)),("three_or_more",collector_count>=3)):
            m=use&mask
            rows[label]={"roots":int(m.sum()),"realized":float(gold[m].mean())}
            for arm,p in pred.items():
                rows[label][arm]={"mean":float(p[m].mean()),"mae":float(np.abs(p[m]-gold[m]).mean())}
        worst=max(roots,key=lambda i:abs(pred["candidate"][i]-gold[i]))
        data={"scope":"Post hoc diagnostic of frozen rival 30-step gross delivery channel; current visible collector count only",
              "predictions_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"groups":rows,
              "largest_absolute_error":roots[worst]}
    (HERE/"rival_freight.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
