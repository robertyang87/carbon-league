"""Map-balanced occupancy learning on frozen original Sapling and Master3s2 source games."""
from collections import Counter
from pathlib import Path
import hashlib
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
TRIAL=HERE.parent/"tree_response_training"
sys.path.insert(0,str(TRIAL/"planter_survival"))
import extract
import forecast

PROTOCOL="6d1bec75712e110ebf089ccf8db31050109fd6ee667c52fa3d1d7df723c9272f"


def check(ok, detail):
    if not ok:
        raise ValueError(detail)


def batch(number):
    check(extract.sha(HERE/"protocol.json")==PROTOCOL,"protocol identity")
    spec=extract.load(HERE/"protocol.json");b=spec["batches"][str(number)]
    path=extract.ROOT/b["source_receipt"];d=path.parent
    check(extract.sha(path)==b["source_sha256"],"source receipt changed")
    signed=extract.load(path)
    original=d/"source/official_replay.json"
    check(extract.sha(original)==signed["official_sha256"]==b["official_sha256"],"signed official source changed")
    if number==5:
        aggregate=d/"aggregate.json"; labels=d/"training_labels.json"
        if not aggregate.exists() or not labels.exists():
            raise FileNotFoundError("waiting for signed full intake05 aggregation")
        a,l=extract.load(aggregate),extract.load(labels)
        check(l["protocol_sha256"]==extract.sha(d/"protocol.json") and len(l["pairs"])==8 and a["complete_pairs"]>=8,"intake05 incomplete")
    official=extract.load(original)["runs"]["source"]
    check(official["pack_sha256"]==spec["parent_sha256"],"wrong v1")
    check(sorted((g["seed"],g["seat"]) for g in signed["games"])==[(m,s) for m in b["maps"] for s in (0,1)],"missing or extra terrain/seat")
    for game in signed["games"]:
        r=next(x for x in official["games"] if (x["seed"],x["seat"])==(game["seed"],game["seat"]))
        check(r["statuses"]==["DONE","DONE"] and r["map_sha256"]==game["map_sha256"] and r["transcript_sha256"]==game["raw_sha256"] and r["official_reward_by_seat"]==game["scores"],"score/rules mismatch")
    return signed["games"]


def distance(a,b):
    dr=abs(a//15-b//15);dc=abs(a%15-b%15)
    return min(dr,15-dr)+min(dc,15-dc)


def load_tape(g, save_states=False):
    p=extract.ROOT/g["raw_path"]
    check(extract.sha(p)==g["raw_sha256"],"raw source changed")
    raw=extract.load(p); cfg=raw["game_configuration"];seat=g["seat"]
    check(not raw["errors"] and raw["pack_sha256"]=="9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0" and raw["examiner_md5"]=="57f9a594cce4f2ec9e3573d8d9d26ca1" and
          len(raw["search_calls"])==g["steps"] and any(r[1]=="searched" for r in raw["search_calls"]), "not verified original full-search source")
    sim=extract.FastCarbon(cfg).reset(seed=g["seed"])
    check(hashlib.sha256(bytes(int(x) for x in sim.carbon)).hexdigest()==g["map_sha256"],"map mismatch")
    observations=[];payments=[];states=[];normal=[]
    for joint in raw["actions"]:
        observations.append(sim.obs(seat))
        if save_states: states.append(sim.get_state())
        before=sim.full_observation()
        after,events=extract.step(sim,before,cfg,joint)
        deposits={e[3] for e in events if e[0] in ("own_delivery","opponent_worker_delivery_to_this_base") and e[2]>0}
        payments.append(deposits)
        normal.append({e[3]:e[2] for e in events if e[0]=="own_delivery" and e[1]==seat})
    observations.append(sim.obs(seat))
    check([round(x,2) for x in sim.cash]==g["scores"],"source official terminal mismatch")
    return raw,observations,payments,states,normal


def sources_at(obs):
    seat=obs["player"];enemy=obs["players"][1-seat]
    home=tuple(enemy[1].values())
    if not home: return []
    return [uid for uid,(c,load,typ) in enemy[2].items() if typ=="COLLECTOR" and load>=120 and min(distance(c,b) for b in home)>=3]


def target(observations,payments,t,uid,h,seat):
    w=observations[t+h]["players"][1-seat][2].get(uid)
    if w is None or w[1]<100 or any(uid in d for d in payments[t:t+h]):
        return -1
    return w[0]


def softmax(z):
    ex=np.exp(z-z.max(axis=-1,keepdims=True))
    return ex/ex.sum(axis=-1,keepdims=True)


def fit():
    groups={3:[],5:[]};labels={3:[],5:[]};maps=[]
    for batch_no in (1,2,3):
        for game in batch(batch_no):
            raw,obs,cash,_,_=load_tape(game);s=game["seat"];cfg=raw["game_configuration"]
            roots=0
            for t in range(len(raw["actions"])-4):
                for uid in sources_at(obs[t]):
                    maps.append(game["seed"]);roots+=1
                    for h in (3,5):
                        cells,x=forecast.target_cells(obs[t],obs[max(0,t-10):t],uid,h,cfg)
                        y=target(obs,cash,t,uid,h,s)
                        check(y in cells and len(set(cells))==len(cells) and -1 in cells, "future position outside registered support")
                        groups[h].append(np.asarray(x,dtype=np.float32))
                        labels[h].append(cells.index(y))
            print("fit features",game["seed"],s,roots,flush=True)
    ids=Counter(maps)
    check(len(ids)==6,"some training terrain missing")
    w=np.array([1./(len(ids)*ids[m]) for m in maps])
    net={"features":list(forecast.NAMES),"horizons":{},"train_map_counts":dict(ids),"loss":{},"protocol_sha256":PROTOCOL}
    for h in (3,5):
        x=np.asarray(groups[h]);y=np.asarray(labels[h]);truth=np.eye(x.shape[1])[y]
        check(np.isfinite(x).all(),"nonfinite forecast features")
        beta=np.zeros(x.shape[-1]);x=x.astype(np.float64)
        for _ in range(100):
            p=softmax(x@beta)
            beta-=.55*(np.einsum("i,ij,ijk->k",w,p-truth,x)+.02*beta)
        p=softmax(x@beta)
        net["horizons"][str(h)]=beta.tolist()
        net["loss"][str(h)]=float(-w@np.log(np.clip(p[np.arange(len(maps)),y],1e-15,1)))
        del x
    net["forecast_sha256"]=extract.sha(HERE/"forecast.py")
    extract.write(HERE/"weights.json",net)
    print("FROZEN",extract.sha(HERE/"weights.json"),ids,net["loss"],flush=True)


if __name__=="__main__":
    fit()
