"""Rebuild one signed, action-aligned example per positive public own cargo decision."""
import gzip
import json
import os
import sys
from collections import deque
from pathlib import Path
import numpy as np
import head
from protocol import HERE,ROOT,TR,sha
sys.path.insert(0,str(TR/"planter_survival"))
from extract import FastCarbon
sys.path.insert(0,str(HERE.parent/"freight_horizon"))
import predictor as old_horizon
from lots import build as timeline_for

CACHE=Path(os.getenv("SAPLING_LOT_CACHE","/tmp/sapling_lot_settlement_20261119"))
def write(path,obj):
    b=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    if path.suffix==".gz":b=gzip.compress(b,mtime=0)
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_bytes(b);os.replace(tmp,path)
def receipt_files(split):
    reg=json.loads((HERE/"protocol.json").read_bytes())
    for part in reg["splits"][split]:
        p=ROOT/part["path"]
        if sha(p)!=part["sha256"]:raise ValueError("source receipt drift")
        signed=json.loads(p.read_bytes())
        for g in signed["games"]:
            path=ROOT/g["raw_path"]
            if sha(path)!=g["raw_sha256"]:raise ValueError("raw source drift")
            yield g,json.loads(gzip.decompress(path.read_bytes()))
def encode_events(events,t):
    n=len(head.CATEGORIES); m=len(head.HORIZONS)
    y=np.zeros(n+2*m,dtype=np.float64); first_ten_unsettled=1.;fraction_paid_later=0.;direct10=0.
    for e in events:
        if not e["is_terminal_allocation"]:continue
        d=e["step"]-t; frac=e["fraction"]; cause=e["kind"]
        if not 1<=d<=299 or frac<0 or cause not in head.CATEGORIES:raise ValueError("wrong event time/cause")
        y[head.CATEGORIES.index(cause)]+=frac
        if d<=10:first_ten_unsettled-=frac
        if cause.startswith("own_"):
            idx=next(i for i,h in enumerate(head.HORIZONS) if d<=h)
            y[n+idx]+=frac
            y[n+m+idx]+=frac*d
            if d>10:fraction_paid_later+=frac
            elif cause=="own_direct" and e["reason"]=="own_bank":direct10+=frac
    if abs(y[:n].sum()-1)>1e-6 or abs(y[:2].sum()-sum(y[n:n+m]))>1e-6:raise ValueError("unnormalized lot flow")
    return y,[max(0.,first_ten_unsettled),fraction_paid_later,direct10]

def build_game(game,raw,cache,old):
    path=cache/f"{game['seed']}_{game['seat']}.npz"
    stamps=dict(raw=game["raw_sha256"],encoder=sha(HERE/"data.py"),account=sha(HERE/"lots.py"),
                inputs=sha(HERE/"head.py"),old=sha(HERE.parent/"freight_horizon/model.json"),old_code=sha(HERE.parent/"freight_horizon/predictor.py"))
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if json.loads(str(z["stamps"]))==stamps:return path
    if len(raw["actions"])!=game["steps"] or not (game["steps"]==299 or "extin" in str(game["termination"]).lower()):
        raise ValueError("truncated trajectory is censored and cannot supply terminal labels")
    timeline=timeline_for(raw,seat=game["seat"],verify=True)
    sim=FastCarbon(raw["game_configuration"]).reset(seed=game["seed"])
    if __import__("hashlib").sha256(bytes(int(x) for x in sim.carbon)).hexdigest()!=game["map_sha256"]:
        raise ValueError("terrain identity")
    s=game["seat"]; history=deque(maxlen=5)
    xx=[];yy=[];xxa=[];keys=[];extra=[];olds=[]
    for t,joint in enumerate(raw["actions"]):
        public=sim.obs(s)
        # Both players' complete worker lists are public. The old enemy model consults
        # those lists and their history only; it does not read the carbon grid here.
        old_obs={"player":1-s,"step":t,"players":public["players"]}
        old_history=[{"player":1-s,"step":h["step"],"players":h["players"]} for h in history]
        old_rows=old.predict(old_obs,old_history)
        for uid,(pos,c,role) in public["players"][s][2].items():
            if role!="COLLECTOR" or c<=0:continue
            if not c==old_rows[uid]["cargo"]:raise ValueError("old model target identity")
            order=joint[s].get(uid) or "NONE"
            x=head.features(public,history,s,uid)
            xa=head.features(public,history,s,uid,order,joint[s])
            y,e=encode_events(timeline.events(t,uid),t)
            xx.append(x);xxa.append(xa);yy.append(y);extra.append(e);keys.append([t,uid,order])
            olds.append(old_rows[uid]["expected_root_cash_10"])
        sim.step(*joint);history.append(public)
    if list(sim.cash)!=game["scores"]:raise ValueError("official terminal cash mismatch")
    tmp=path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,x=xx,xa=xxa,y=yy,extra=extra,old=olds,keys=keys,stamps=json.dumps(stamps))
    os.replace(tmp,path)
    return path

def generate(split):
    CACHE.mkdir(exist_ok=True,parents=True)
    old=old_horizon.FreightPredictor(HERE.parent/"freight_horizon/model.json");receipt=[]
    for game,raw in receipt_files(split):
        path=build_game(game,raw,CACHE,old)
        with np.load(path,allow_pickle=False) as z:
            receipt.append(dict(seed=game["seed"],seat=game["seat"],terrain=game["map_sha256"],tape=game["raw_sha256"],npz_sha256=sha(path),
                                file=path.name,roots=len(z["x"]),terminal_cash=game["scores"],turns=game["steps"]))
    write(HERE/(split+"_sources.json"),receipt)
    return receipt
if __name__=="__main__":generate(sys.argv[1])
