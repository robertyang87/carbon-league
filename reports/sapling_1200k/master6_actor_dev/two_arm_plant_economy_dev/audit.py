"""Independent joins and probability check, including evidence by match result and physical seat."""
from collections import defaultdict
import gzip
import json
import numpy as np
from scipy.special import expit
from plan import HERE,PRE,read,save,sha,TOP
from economic import _predict,NAMES

def main():
    plan=read(HERE/"plan.json");maps=plan["maps"];base=HERE.parent/"planter_intent_labels"
    prev=read(base/"verification.json")
    for k,h in {**prev["shard_sha256"],**prev["source_sha256"]}.items():assert sha(base/k)==h
    receipt=read(HERE/"label_receipt.json")
    assert receipt["plan_sha256"]==sha(HERE/"plan.json") and receipt["features_sha256"]==sha(HERE/"economic.py")
    old={m["terrain"] for m in read(PRE/"panel.json")["maps"]}|{m["map"] for m in read(PRE/"parent_panel/panel.json")["games"]}
    assert not old&{m["terrain"] for m in maps}
    assert len(maps)==len({m["terrain"] for m in maps})==len({m["seed"] for m in maps})
    opponents={r["opponent"] for r in maps if r["split"] in ("train","calibration")}
    assert opponents.isdisjoint(r["opponent"] for r in maps if r["split"]=="new_opponent")
    model=read(HERE/"model.json");assert model["features"]==NAMES
    scores={};counters=defaultdict(list);examples=[]
    for n,row in enumerate(maps):
        if row["split"] not in ("new_opponent","known_new_map"):continue
        mf=row["source"]
        if mf not in scores:scores[mf]={r["id"]:r["score_by_seat"] for r in read(TOP/mf)["games"]}
        pts=scores[mf][row["id"]];s=row["seat"]
        outcome="won" if pts[s]>pts[1-s] else "lost" if pts[s]<pts[1-s] else "drawn"
        src=HERE/"labels"/receipt["files_in_map_order"][n];assert sha(src)==receipt["source_sha256"][n]
        with np.load(src,allow_pickle=False) as z:
            x,y,o=z["x"],z["y"],z["ident"]
        p=np.zeros(len(x));nval=p.copy();pq=p.copy();pn=p.copy()
        for arm,select in (("stay",o[:,4]==0),("move",o[:,4]!=0)):
            w=model["arms"][arm];ds,di=w["death_cal"];cs,ci=w["net_cal"]
            p[select]=expit(ds*np.asarray([_predict(w["death"],v) for v in x[select]])+di)
            nval[select]=cs*np.asarray([_predict(w["net"],v) for v in x[select]])+ci
            pq[select]=w["baseline_death"];pn[select]=w["baseline_net"]
        metrics=[len(x),np.mean((y[:,0]-p)**2),np.mean((y[:,0]-pq)**2),np.mean(np.abs(y[:,1]-nval)),np.mean(np.abs(y[:,1]-pn))]
        for group in ("all",outcome,"seat"+str(s),row["opponent"]):counters[(row["split"],group)].append(metrics)
    stats={}
    for (spl,group),rows in counters.items():
        vals=np.asarray(rows);mean=vals.mean(axis=0);g=np.random.default_rng(771540);ix=g.integers(0,len(vals),(4000,len(vals)))
        boot=vals[ix].mean(axis=1)
        def interval(j,k):return np.quantile(boot[:,j]-boot[:,k],[.025,.975]).tolist()
        stats.setdefault(spl,{})[group]={"maps":len(vals),"rows":int(vals[:,0].sum()),"brier":mean[1],
            "brier_diff":mean[1]-mean[2],"brier_diff_map95":interval(1,2),
            "net_mae":mean[3],"net_mae_diff":mean[3]-mean[4],"net_mae_diff_map95":interval(3,4)}
    evidence={"verified_prior_shards":len(prev["shard_sha256"]),"model_sha256":sha(HERE/"model.json"),
        "label_receipt_sha256":sha(HERE/"label_receipt.json"),"prior_maps_overlap":0,
        "by_split_outcome_physical_seat_and_opponent":stats,
        "interval":"Conditional on this fitted model, resample held physical maps. Each group/metric uses map means; one map cannot give useful between-map uncertainty."}
    save(HERE/"audit.json",evidence)
if __name__=="__main__":main()
