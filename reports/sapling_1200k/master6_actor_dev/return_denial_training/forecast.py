"""Fit a next-command distribution on the two training maps and score one physical interference."""
from collections import Counter, deque
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
TRIAL=HERE.parent/"tree_response_training"
sys.path.insert(0,str(TRIAL/"planter_survival"))
import extract
from policy import MOVES, NAMES, public_turn

PROTOCOL="dfb98ea08e28b61b9eb8606d9918b56ada55d1010066ed48b6cf03622dfe709d"


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def bind(split):
    require(extract.sha(HERE/"protocol.json")==PROTOCOL,"unregistered protocol")
    p=extract.load(HERE/"protocol.json")
    s=p["split"][split]
    b=TRIAL/"batches"/("intake_"+s["batch"])
    require(extract.sha(b/"source_receipt.json")==s["receipt_sha256"] and
            extract.sha(b/"source/official_replay.json")==s["official_sha256"],"unregistered source receipts")
    signed=extract.load(b/"source/official_replay.json")["runs"]["source"]
    require(signed["pack_sha256"]==p["parent_pack_sha256"],"original actor identity mismatch")
    for g in s["tapes"]:
        r=next(r for r in signed["games"] if (r["seat"],r["seed"])==(g["seat"],g["seed"]))
        require(r["statuses"]==["DONE","DONE"] and r["transcript_sha256"]==g["raw_sha256"] and
                r["official_reward_by_seat"]==g["scores"] and r["map_sha256"]==g["map_sha256"],"source cells or terminal mismatch")
    return s


def restore(g):
    path=extract.ROOT/g["raw_path"]
    require(extract.sha(path)==g["raw_sha256"],"wrong source tape")
    raw=extract.load(path)
    require(raw["candidate_seat"]==g["seat"] and raw["seed"]==g["seed"] and len(raw["actions"])==g["steps"],"source tape identity")
    cfg=raw["game_configuration"]
    sim=extract.FastCarbon(cfg).reset(seed=g["seed"])
    import hashlib
    require(hashlib.sha256(bytes(int(v) for v in sim.carbon)).hexdigest()==g["map_sha256"],"map identity")
    return raw,cfg,sim


def softmax(z):
    e=np.exp(z-np.max(z,axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)


def training():
    spec=bind("train")
    features=[]; labels=[]; maps=[]; activity=Counter()
    for g in spec["tapes"]:
        raw,cfg,sim=restore(g)
        seat=g["seat"]
        history=deque(maxlen=10)
        for t,joint in enumerate(raw["actions"]):
            obs=sim.obs(seat)
            public=public_turn(obs,list(history),joint[seat],cfg)
            for target in public:
                label=joint[1-seat].get(target["target"]) or "NONE"
                features.append(target["features"]); labels.append(MOVES.index(label)); maps.append(g["seed"]);activity[label]+=1
            sim.step(*joint)
            history.append(obs)
        require([round(x,2) for x in sim.cash]==g["scores"],"training tape terminal mismatch")
    counts=Counter(maps)
    require(set(counts)==set(spec["maps"]),"a training map yielded no returning carriers")
    x=np.asarray(features,dtype=float); y=np.asarray(labels)
    require(x.shape==(len(labels),5,len(NAMES)) and np.isfinite(x).all(),"feature shape or NaN")
    w=np.asarray([1/(len(counts)*counts[m]) for m in maps])
    truth=np.eye(5)[y]
    beta=np.zeros(len(NAMES))
    for _ in range(100):
        p=softmax(x@beta)
        gradient=np.einsum("i,ij,ijk->k",w,p-truth,x)+.02*beta
        beta-=.7*gradient
    p=softmax(x@beta)
    doc=dict(protocol_sha256=PROTOCOL,feature_code_sha256=extract.sha(HERE/"policy.py"),features=list(NAMES),weights=beta.tolist(),
             fit_map_counts=dict(counts),training_action_counts=dict(activity),training_map_equal_logloss=float(-w@np.log(p[np.arange(len(y)),y])),
             iterations=100,learning_rate=.7,l2=.02)
    extract.write(HERE/"weights.json",doc)
    print("frozen forecast",extract.sha(HERE/"weights.json"),counts,flush=True)


def select(carriers,model,cfg):
    """Use only scored public carriers. Candidate and prior share a fixed physical set."""
    by_type={"forecast":[],"geometry":[]}
    for c in carriers:
        require(model["features"]==list(NAMES),"model mismatch")
        p=softmax(np.asarray(c["features"])@np.asarray(model["weights"])).tolist()
        shortest=np.array(c["shortest"],dtype=float)
        geom=(shortest/shortest.sum()).tolist()
        c["p"]=p
        for label,probs in (("forecast",p),("geometry",geom)):
            by_cell=dict(zip(c["action_cells"],probs))
            for o in c["options"]:
                gain=c["cargo"]*(1-cfg["moveCost"])*(by_cell[o["cell"]]-by_cell.get(o["old_cell"],0))
                key=(-gain,c["target"],o["own_uid"],MOVES.index(o["order"]))
                by_type[label].append((key,c,o,gain))
    return {label:min(options,key=lambda z:z[0])[1:] if options else None for label,options in by_type.items()}


if __name__=="__main__":
    training()
