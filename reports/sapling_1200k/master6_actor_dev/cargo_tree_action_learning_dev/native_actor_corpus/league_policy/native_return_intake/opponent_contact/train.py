"""Fit one full-history model and its prespecified geometry-only control on four maps."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import features
import model


HERE=Path(__file__).resolve().parent
PLAN="2c904de2e1c31c49a9d8897123f0569e71728896b46da9f792ccbdb0b9e3c18c"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_map_weights(seed,episode):
    w=np.zeros(len(seed),dtype=np.float64)
    for s in np.unique(seed):
        games=np.unique(episode[seed==s])
        for ep in games:
            subset=(seed==s)&(episode==ep)
            w[subset]=1/(len(np.unique(seed))*len(games)*int(subset.sum()))
    if not math.isclose(math.fsum(w),1.,abs_tol=1e-12):
        raise ValueError("invalid per-expedition design weights")
    return w


def persistence(x):
    lag=x[:,features.HISTORY_OFFSET:features.HISTORY_OFFSET+6]
    if not np.all(lag.sum(1)==1):
        raise ValueError("missing previous-action indicator")
    prev=lag.argmax(1)
    out=np.full((len(x),5),.05)
    for action in range(5):
        out[prev==action,action]=.8
    out[prev==5]=.2
    return out


def binary_metrics(p,y):
    n=len(y)
    metrics={"opportunities":n,"events":int(y.sum()),"mean_prediction":float(p.mean()) if n else None,
             "brier":float(np.mean((p-y)**2)) if n else None}
    for threshold in (.2,.5):
        forecast=p>=threshold
        fp=int(np.sum(forecast & ~y));fn=int(np.sum(~forecast & y))
        neg=int((~y).sum());pos=int(y.sum())
        metrics[str(threshold)]={"tp":pos-fn,"fp":fp,"tn":neg-fp,"fn":fn,
            "false_positive_rate":fp/neg if neg else None,"false_negative_rate":fn/pos if pos else None}
    return metrics


def movement_metrics(p,y):
    chosen=p[np.arange(len(p)),y]
    return {"opportunities":len(y),"top1":float(np.mean(p.argmax(1)==y)),
            "brier":float(np.mean((p*p).sum(1)-2*chosen+1)),
            "log_loss":float(-np.mean(np.log(chosen)))}


def contact(p,d):
    counts=np.diff(d['link_ptr'])
    owner=np.repeat(np.arange(len(counts)),counts)
    # Product is well defined for zero links (exactly zero contact) or p=1.
    safe=np.ones(len(counts))
    np.multiply.at(safe,owner,1-p[d['link_idx'],d['link_action']])
    return 1-safe


def evaluate(weights,data,held):
    seed,ep,goal,move=data["seed"],data["ep"],data["g_seed"],data["y"]
    probs={kind:model.predict(w,data["x"]) for kind,w in weights.items()}
    probs["persistence"]=persistence(data["x"])
    ptr=data["link_ptr"]
    links=np.diff(ptr)
    check_y=np.zeros(len(links),bool)
    np.logical_or.at(check_y,np.repeat(np.arange(len(links)),links),
                     move[data["link_idx"]]==data["link_action"])
    if not np.array_equal(check_y,data["g_truth"].astype(bool)):
        raise ValueError("link roles disagree with recorded intent event")
    target_intent=data["g_intent"]
    masks={"all":np.ones(len(links),bool),"reachable":links>0,"stay":target_intent==0,"move":target_intent!=0,
           "reachable_stay":(links>0)&(target_intent==0),"reachable_move":(links>0)&(target_intent!=0)}
    result={}
    for which in ("pooled",*held):
        action_rows=np.isin(seed,held) if which=="pooled" else seed==which
        goal_rows=np.isin(goal,held) if which=="pooled" else goal==which
        entry={"episodes":int(len(np.unique(ep[action_rows]))),"models":{}}
        for name,p in probs.items():
            q=contact(p,data)
            entry["models"][name]={"movement":movement_metrics(p[action_rows],move[action_rows]),
                "contact":{key:binary_metrics(q[mask&goal_rows],check_y[mask&goal_rows])
                           for key,mask in masks.items()}}
        result[str(which)]=entry
    mean={}
    for name in probs:
        cards=[result[str(s)]["models"][name] for s in held]
        mean[name]={"movement":{k:float(np.mean([r["movement"][k] for r in cards]))
                                for k in ("brier","top1","log_loss")},
                    "contact_brier":{key:float(np.mean([r["contact"][key]["brier"] for r in cards])) for key in masks}}
    result["equal_map_mean"]=mean
    delta={str(s):round(result[str(s)]["models"]["full"]["contact"]["reachable"]["brier"]-
                        result[str(s)]["models"]["geometry"]["contact"]["reachable"]["brier"],12) for s in held}
    result["prespecified_contact_decision"]={"reachable_brier_full_minus_geometry_by_map":delta,
                          "qualified_for_separate_ablation":all(x<0 for x in delta.values()),
                          "deployed":False}
    return result


def train():
    plan=json.loads((HERE/"protocol.json").read_text())
    if sha(HERE/"protocol.json")!=PLAN:
        raise ValueError("unexpected preregistration")
    path=HERE/"dataset.npz"
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    fit=np.isin(data['seed'],plan['fit_terrain_seeds'])
    held=plan['heldout_terrain_seeds']
    if not np.array_equal(fit,data['is_train'].astype(bool)) or set(np.unique(data["seed"][~fit]))!=set(held):
        raise ValueError("wrong held-out maps")
    if len(np.unique(data["ep"][fit]))!=16 or len(np.unique(data["ep"][~fit]))!=8:
        raise ValueError("incomplete episode split")
    trained=model.fit(data["x"][fit],data["y"][fit],
                      physical_map_weights(data["seed"][fit],data["ep"][fit]),
                      features.HISTORY_COLS,features.CONTINUOUS_COLS)
    for kind,weights in trained.items():
        np.savez(HERE/f"{kind}.npz",**weights)
    scores=evaluate(trained,data,held)
    provenance={f:sha(HERE/f) for f in ("protocol.json","dataset.py","features.py","model.py",
                                         "train.py","dataset.json","dataset.npz","full.npz","geometry.npz")}
    card={"schema":"opponent-contact-supervised-v1", "files_sha256":provenance,
          "feature_count":len(features.NAMES),"geometry_features":features.HISTORY_OFFSET,
          "learned_parameters":32*len(features.NAMES)+32+32*5+5,
          "fit_worker_rows":int(fit.sum()),"held_worker_rows":int((~fit).sum()),"metrics":scores}
    (HERE/"results.json").write_text(json.dumps(card,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return scores["prespecified_contact_decision"]


if __name__=="__main__":
    print(json.dumps(train()))
