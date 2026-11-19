"""Reuse recorded decisions and verified purchase ledgers; open only predeclared replay maps for public movement vectors."""
import gzip, hashlib, json, tarfile
from pathlib import Path
from collections import defaultdict
import numpy as np
from pydantic_core import from_json
import economic
from plan import HERE,PRE,read,sha,TOP
import sys
sys.path.insert(0,str(PRE))
from registry import save,REPLAYS
DATA=HERE.parent/"planter_intent_labels"

def purchases(r):
    if r["source"]=="manifest.json":
        p=TOP/"plant_timing_data"/(r["id"][0]+".tar")
        with tarfile.open(p) as tar: raw=from_json(gzip.decompress(tar.extractfile(r["id"]+".json.gz").read()))
        assert raw["replay_sha256"]==r["raw_sha256"]
        return raw["purchases"]
    p=TOP/Path(r["source"]).parent/"cases.json.gz"
    row=next(x for x in read(p)["rows"] if x["id"]==r["id"])
    assert row["source_sha256"]==r["raw_sha256"]
    return row["planting"]["purchases"]

def main():
    maps=read(HERE/"plan.json")["maps"]
    archive=read(DATA/"maps.json.gz")["maps"]
    coverage=read(DATA/"verification.json")
    assert sha(DATA/"maps.json.gz")==coverage["source_sha256"]["maps.json.gz"]
    index={r["id"]:r["map_index"] for r in archive}
    selected={index[r["id"]]:(i,r) for i,r in enumerate(maps)}
    directory=HERE/"labels";directory.mkdir(exist_ok=True)
    for shard in sorted(DATA.glob("labels_*.npz")):
        with np.load(shard,allow_pickle=False) as z:
            mi=z["map_index"]
            ix=np.flatnonzero(np.isin(mi,list(selected)))
            if not len(ix):continue
            values={k:z[k] for k in ("seat","affordable","action","t","cell","worker_number","y_survived","y_paid","y_actual_price")}
            for m in sorted(set(mi[ix])):
                out=directory/(maps[selected[m][0]]["id"]+".npz")
                if out.exists():continue
                mapno,card=selected[m];seat=card["seat"];p=REPLAYS/(card["id"]+".json.gz")
                assert sha(p)==card["raw_sha256"]
                game=from_json(gzip.decompress(p.read_bytes()))
                turns=game["steps"];cfg=game["configuration"]
                ledger=purchases(card); rents={(q[0],q[1],q[2],q[4]):q for q in ledger}
                assert len(rents)==len(ledger)
                X=[]; labels=[]; ident=[]
                for j in ix[(mi[ix]==m)&(values["seat"][ix]==seat)&(values["affordable"][ix]==1)]:
                    t=int(values["t"][j]);cmdno=int(values["action"][j]);cell=int(values["cell"][j]);wid=int(values["worker_number"][j])
                    uid=f"player-{seat}-worker-{wid}"
                    obs=economic.public(turns[t],seat)
                    history=[economic.public(turns[l],seat) for l in range(max(0,t-3),t)]
                    joint=turns[t+1][seat]["action"] or {}
                    act=joint.get(uid) or "NONE"
                    assert cmdno==("NONE","UP","RIGHT","DOWN","LEFT").index(act) and obs['players'][seat][2][uid][0]==cell
                    x,fee=economic.features(obs,history,joint,uid,cfg)
                    dest=economic.geom.destination(cell,act)
                    buy_t=t+(cmdno!=0)
                    q=rents.get((buy_t,seat,uid,dest))
                    alive=int(values["y_survived"][j]);assert alive==int(uid in turns[t+1][0]["observation"]["players"][seat][2])
                    if cmdno==0:assert bool(q)==bool(values["y_paid"][j])
                    if q:assert alive and (cmdno!=0 or abs(float(values["y_actual_price"][j])-q[6])<.000051)
                    net=q[11]+q[13]-q[6] if q else 0.
                    X.append(x);labels.append([1-alive,net,int(q is not None),q[11]+q[13] if q else 0.,q[6] if q else 0.])
                    ident.append([t,wid,cell,dest,cmdno])
                partial=out.with_suffix('.partial')
                with partial.open('wb') as f:
                    np.savez_compressed(f,x=np.asarray(X,np.float32).reshape(-1,len(economic.NAMES)),
                        y=np.asarray(labels,np.float64).reshape(-1,5),ident=np.asarray(ident,np.int16).reshape(-1,5))
                partial.replace(out)
    files=[directory/(r["id"]+".npz") for r in maps]
    save(HERE/"label_receipt.json",{"plan_sha256":sha(HERE/"plan.json"),"features_sha256":sha(HERE/"economic.py"),
        "source_verification_sha256":sha(DATA/"verification.json"),"source_sha256":[sha(p) for p in files],"feature_names":economic.NAMES,
        "y_columns":["died_on_root","first_tenure_net","purchased","first_tenure_gross","actual_tree_fee"],
        "identity":["step","worker_number","start_cell","requested_destination","proposed_action"],"files_in_map_order":[p.name for p in files]})
    print(len(files),flush=True)

if __name__=="__main__":main()
