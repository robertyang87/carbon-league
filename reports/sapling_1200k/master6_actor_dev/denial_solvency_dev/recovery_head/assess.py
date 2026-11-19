"""Blind held-map evaluation from immutable deployed weights; all rates and errors give each map equal weight."""
from collections import defaultdict
import gzip
import json
import numpy as np
from dataset import HERE, fetch
from fit import read_rows, map_weights,cell
from head import predict

MODELS=("constant","tree_gap_table","tree_gap","state")


def assess(rows):
    if not rows:return {"events":0,"maps":0}
    w=map_weights(rows)
    y=np.array([r["label_30"] for r in rows])
    output={"events":len(rows),"maps":len({r["map"] for r in rows}),"positive":sum(r["label_30"] for r in rows),
            "map_equal_rate":float(w@y),"models":{}}
    for m in MODELS:
        p=np.array([r["predictions"][m] for r in rows])
        bins=np.minimum(4,(p*5).astype(int))
        logloss=-(y*np.log(p)+(1-y)*np.log1p(-p))
        calibration=[]
        for b in range(5):
            take=bins==b
            if not any(take):
                calibration.append({"interval":[b/5,(b+1)/5],"events":0});continue
            v=w[take]
            observed=float(np.average(y[take],weights=v));pred=float(np.average(p[take],weights=v))
            calibration.append({"interval":[b/5,(b+1)/5],"events":int(sum(take)),"maps":len({r["map"] for r,ok in zip(rows,take) if ok}),
                                "total_map_weight":float(sum(v)),"observed":observed,"predicted":pred})
        output["models"][m]={"Brier":float(w@((p-y)**2)), "log_loss":float(w@logloss),"mean_prediction":float(w@p),
                             "calibration":calibration,
                             "fixed_bin_ece":sum(c["total_map_weight"]*abs(c["predicted"]-c["observed"]) for c in calibration if c["events"])}
    return output


def main():
    model=fetch.load(HERE/"weights.json")
    if fetch.sha((HERE/"split.json").read_bytes())!=model["split_sha256"]:raise ValueError("fit belongs to different split")
    rows=read_rows()
    for r in rows:
        r["predictions"]={"constant":model["constant"],"tree_gap_table":model["tree_gap_table"][cell(r)],
                          **{k:predict(r["public"],model[k]) for k in ["state","tree_gap"]}}
    test=[r for r in rows if r["split"]!="train" and r["label_30"] is not None]
    groups={"main":test,"train":[r for r in rows if r["split"]=="train" and r["label_30"] is not None],
            "opponent_unseen_train":[r for r in test if not r["rival_seen_in_fit"]],
            "opponent_seen_train":[r for r in test if r["rival_seen_in_fit"]],
            "seat0":[r for r in test if r["seat"]==0],"seat1":[r for r in test if r["seat"]==1]}
    for state in "WDL":groups["actor_"+state]=[r for r in test if r["result"]==state]
    for role in ("Master6","other_top"):
        groups[role]=[r for r in test if r["actor"].endswith("Master6")== (role=="Master6")]
    scores={k:assess(v) for k,v in groups.items()}
    maps=sorted({r["map"] for r in test})
    bymap=[assess([r for r in test if r["map"]==b])["models"] for b in maps]
    ix=np.random.default_rng(550099).integers(0,len(maps),size=(2000,len(maps)))
    intervals={}
    for base in ("constant","tree_gap_table","tree_gap"):
        intervals[base]={}
        for name in ("Brier","log_loss"):
            per=np.array([x["state"][name]-x[base][name] for x in bymap])
            intervals[base][name]=np.quantile(per[ix].mean(axis=1),[.025,.975]).tolist()
    def evidence(r):
        return {k:r[k] for k in ("game","pre_t","seat","actor","rival","result","rival_seen_in_fit","label_30","public","predictions")}
    tails=[evidence(r) for r in sorted(test,key=lambda r:-abs(r["predictions"]["state"]-r["label_30"]))[:5]]
    val={"weights_sha256":fetch.sha((HERE/"weights.json").read_bytes()),"scored_maps_both_corpora":1528,"all_realized_trades":len(rows),
         "censored":sum(r["label_30"] is None for r in rows),"groups":scores,"paired_map_bootstrap_difference_state_minus_control":intervals,
         "largest_absolute_errors":tails}
    fetch.dump(HERE/"evaluation.json",val)
    export=[{k:r[k] for k in ("game","map","pre_t","seat","actor","rival","rival_seen_in_fit","split","label_30","predictions")} for r in rows]
    fetch.atomic(HERE/"predictions.jsonl.gz",gzip.compress(b"".join((json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n").encode() for r in export),mtime=0))

if __name__=="__main__":main()
