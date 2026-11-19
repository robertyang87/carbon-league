"""Fit a victim's simultaneous escape/return distribution, with map and opponent holdouts."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize

from pressure_return_features import pressure_features,_PRESSURE_DIST,_PRESSURE_FEATURES
HERE=Path(__file__).resolve().parent;ALL=HERE/"all_top5";OUT=HERE/"all_top5_pressure_dev"
sys.path.insert(0,str(ALL))
from raid_timing import game_rows

def sha(b):return hashlib.sha256(b).hexdigest()
def bucket(s,n):return int(sha(s.encode())[:16],16)%n
def save(p,obj):
    t=p.with_suffix(".tmp");t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n");t.replace(p)


def main(cache):
    man=json.loads((ALL/"manifest.json").read_bytes());games=man["games"]
    app=json.loads((ALL/"appendix_499886f1/manifest.json").read_bytes())["games"]
    old=json.loads((HERE/"latest5_inventory.json").read_bytes())["games"]
    excluded={g["seed"] for g in old};old_ids={g["id"] for g in old}
    xfile=ALL/"raid_timing_rows.npz";orig=np.load(xfile)["rows"]
    ref=json.loads(gzip.decompress((ALL/"raid_timing.json.gz").read_bytes()))
    if sha(xfile.read_bytes())!=ref["rows_sha256"] or [r["id"] for r in ref["games"]]!=[g["id"] for g in games]:
        raise ValueError("opportunity input order or digest differs")
    first=json.loads(gzip.decompress((cache/games[0]["path"]).read_bytes()))
    bases=[list(q[1].values()) for q in first["steps"][0][0]["observation"]["players"]]
    hd=np.array([_PRESSURE_DIST[:,b].min(axis=1) for b in bases])
    before=len(games); parts=[orig]; extra_seeds={r["seed"] for r in games}
    for r in app:
        if r["seed"] in extra_seeds or r["seed"] in excluded or r["id"] in old_ids:continue
        p=cache/r["path"];raw=p.read_bytes();data=json.loads(gzip.decompress(raw))
        if sha(raw)!=r["sha256"] or data["statuses"]!=["DONE","DONE"]:raise ValueError("appendix archive mismatch")
        parts.append(game_rows(len(games),data,{"lots":[],"enemy_base_refunds":[]}))
        games.append(r);extra_seeds.add(r["seed"])
    ar=np.concatenate(parts);groups={}; assignments=[];identity_by_name={}
    for g in games:
        for n,i in zip(g["seat_names"],g["seat_strategy_ids"]):
            if i:identity_by_name.setdefault(" ".join(n.casefold().split()),set()).add(i)
    def victim(g,side):
        name=" ".join(g["seat_names"][side].casefold().split()); ids=identity_by_name.get(name,set())
        return g["seat_strategy_ids"][side] or (next(iter(ids)) if len(ids)==1 else "name:"+name)
    if not np.array_equal(hd[1-ar["seat"],ar["victim_cell"]],ar["enemy_home"]):raise ValueError("base geometry mismatch")
    master=man["top5"][1]["id"]
    rng=np.random.default_rng(982393)
    offsets=np.searchsorted(ar["game"],np.arange(len(games)+1))
    for idx,g in enumerate(games):
        if g["id"] in old_ids or g["seed"] in excluded:continue
        own=ar[offsets[idx]:offsets[idx+1]]
        map_bin=bucket("pressure-return-map/"+str(g["seed"]),10)
        entries={}
        for s in (0,1):
            who=victim(g,1-s); held=(who!=master and bucket("pressure-return-victim/"+who,5)==0)
            group=("appendix" if idx>=before else
                   ("train" if not held else "excluded_victim") if map_bin<7 else
                   ("val_new_victim" if held else "val_known_victim") if map_bin==7 else
                   ("test_new_victim" if held else "test_known_victim"))
            entries[str(1-s)]={"victim":who,"set":group}
            if group=="excluded_victim":continue
            inds=np.flatnonzero(own["seat"]==s)+offsets[idx]
            groups.setdefault((group,g["seed"]),[]).extend(inds.tolist())
        assignments.append({"id":g["id"],"map":g["seed"],"victim_seats":entries})
    save(OUT/"split.json",{"maps":assignments,"excluded_prior_26":sorted(excluded),"original_rows":len(orig),"appendix_games":len(games)-before})
    datasets={}
    for (group,seed),indices in sorted(groups.items()):
        if not indices:continue
        hit=rng.choice(indices,size=min(80,len(indices)),replace=False)
        datasets.setdefault(group,[]).append((hit,seed))
    def table(group):
        chunks=datasets.get(group,[])
        inds=np.concatenate([i for i,_ in chunks]);rows=ar[inds]
        features=pressure_features(rows["hunter_cell"],rows["victim_cell"],rows["their_cargo"],rows["enemy_last_move"],rows["enemy_toward_home"],rows["step"],hd[1-rows["seat"]])
        weights=np.concatenate([np.full(len(i),1./len(i)/len(chunks)) for i,_ in chunks])
        return features,rows["their_action"],weights,len(chunks)
    X,y,w,_=table("train"); rows=np.arange(len(y))
    def loss(v):
        z=X@v;z-=z.max(1,keepdims=True); ex=np.exp(z);p=ex/ex.sum(1,keepdims=True)
        total=-np.dot(w,np.log(p[rows,y]))+.01*np.dot(v,v)
        p[rows,y]-=1
        grad=((X*p[:,:,None]).sum(1)*w[:,None]).sum(0)+.02*v
        return total,grad
    fit=minimize(loss,np.zeros(X.shape[-1]),jac=True,method="L-BFGS-B",options={"maxiter":90,"ftol":1e-13})
    if not fit.success or not np.isfinite(fit.x).all():raise RuntimeError(str(fit.message))
    def measures(a,y,w,nmap,coef):
        z=a@coef;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);target=np.eye(5)[y]
        return {"rows":len(y),"maps":nmap,"log_loss":float(-w@np.log(p[np.arange(len(y)),y])),
                "brier":float(w@((p-target)**2).sum(1)),"top1":float(w@(p.argmax(1)==y))}
    stats={}
    for name in datasets:
        a,y2,w2,n=table(name);greedy=np.zeros(len(fit.x));greedy[1]=1.
        stats[name]={"fitted":measures(a,y2,w2,n,fit.x),"home_direction_only":measures(a,y2,w2,n,greedy),
                     "uniform_log_loss":float(np.log(5))}
    save(OUT/"model.json",{"features":_PRESSURE_FEATURES,"coefficients":fit.x.tolist(),"metrics":stats,
          "source_npz_sha256":sha(xfile.read_bytes()),"split_sha256":sha((OUT/"split.json").read_bytes()),
          "plan_sha256":sha((OUT/"plan.json").read_bytes()),"regularization":.02,"optimizer_iterations":fit.nit,
          "original_geometry_checked_rows":len(ar),"equal_map_weight":True})
    print({k:(v["fitted"]["rows"],v["fitted"]["log_loss"],v["home_direction_only"]["log_loss"]) for k,v in stats.items()},flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True,type=Path);main(p.parse_args().cache)
