"""One pre-registered parent-anchored full-batch optimization; test is invoked separately."""
import argparse
import gzip
import json
import datetime as dt
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from register import H, sha, save, get
from data import extract
from model import Critic, module, parent, parse, package, weight_blob

PROTO = "e162d26dfbbcc2118d02450046fe330796212f7ed22aec826bfe2467ba147b3a"


def file(path,data):
    if path.exists() and path.read_bytes()!=data:
        raise ValueError(f"Conflicting output {path}")
    path.write_bytes(data)


def prep(split):
    h,s,rows,w,fixture=extract(split)
    h,s,w=map(torch.from_numpy,(h,s,w))
    y=torch.tensor([r["truth"] for r in rows],dtype=torch.float32)
    model=Critic(parse(weight_blob(parent())))
    with torch.no_grad():
        base=torch.cat([model(h[i:i+32],s[i:i+32]) for i in range(0,len(h),32)])
    return h,s,rows,w,fixture,y,model,base


def fit():
    h,s,rows,w,fixture,y,model,base=prep("fit")
    opt=torch.optim.Adam(model.parameters(),lr=.0003)
    losses=[]
    for epoch in range(40):
        opt.zero_grad(set_to_none=True)
        total=0.
        for i in range(0,len(h),32):
            j=slice(i,i+32)
            value=model(h[j],s[j])
            target=F.huber_loss(value,y[j],delta=.75,reduction="none")
            loss=((target + .5*(value-base[j]).square())*w[j]).sum()
            loss.backward()
            total+=loss.item()
        opt.step()
        losses.append(total)
    blob=model.as_blob()
    file(H/"weights.npz",blob)
    file(H/"candidate.py",package(blob))
    learned=parse(blob); old=parse(weight_blob(parent()))
    fit_path=H/"fit.json"
    frozen_at=get(fit_path)["freeze_utc"] if fit_path.exists() else dt.datetime.now(dt.timezone.utc).isoformat()
    return save(fit_path,dict(protocol=PROTO,weights_sha256=sha(blob),candidate_sha256=sha(package(blob)),
                                  fit_roots=len(rows),fit_games=len({r["game"] for r in rows}),
                                  fit_maps=len({r["terrain"] for r in rows}),objective_per_epoch=losses,
                                  updated={k:dict(shape=list(v.shape),count=v.size,changed=int((v!=old[k]).sum()),
                                      max_absolute=float(np.abs(v-old[k]).max())) for k,v in learned.items()},
                                  freeze_utc=frozen_at))


def grade(sub):
    maps=defaultdict(list)
    for r in sub:
        maps[r["terrain"]].append(r)
    def calc(rs,k):
        w=np.array([r["weight"] for r in rs]);w=w/w.sum()
        pred=np.array([r[k] for r in rs]);y=np.array([r["truth"] for r in rs])
        return np.array([np.sum(w*np.abs(pred-y)),np.sum(w*(pred-y)**2),np.sum(w*(pred-y)),
                         np.sum(w*(np.sign(pred)==np.sign(y)))])
    metric_names=["MAE","MSE","bias","sign_accuracy"]
    by_map={t:{k:calc(rs,k).tolist() for k in ["parent","new","cash_terminal"]} for t,rs in maps.items()}
    all_stats={k:dict(zip(metric_names,np.mean([m[k] for m in by_map.values()],axis=0).tolist()))
               for k in ["parent","new","cash_terminal"]}
    d=np.array([v["new"][0]-v["parent"][0] for v in by_map.values()])
    boot=np.random.default_rng(487113).choice(d,(10000,len(d)),replace=True).mean(1)
    bins={}
    for k in ["parent","new"]:
        bins[k]=[]
        for a,b in zip([-1e9,-1,-.5,0,.5,1],[-1,-.5,0,.5,1,1e9]):
            rs=[r for r in sub if a<=r[k]<b]
            if not rs:
                bins[k].append(dict(lower=a,upper=b,roots=0));continue
            weight=np.array([r["weight"] for r in rs])
            bins[k].append(dict(lower=a,upper=b,roots=len(rs),maps=len({r["terrain"] for r in rs}),
                forecast=float(np.average([r[k] for r in rs],weights=weight)),
                outcome=float(np.average([r["truth"] for r in rs],weights=weight))))
    return dict(roots=len(sub),games=len({r["game"] for r in sub}),maps=len(maps),estimates=all_stats,
        mae_delta_map_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),maps_improved=int((d<0).sum()),
        metric_order=metric_names,per_map=by_map,calibration=bins)


def test():
    trained=get(H/"fit.json")
    blob=(H/"weights.npz").read_bytes()
    assert sha(blob)==trained["weights_sha256"] and sha((H/"candidate.py").read_bytes())==trained["candidate_sha256"]
    h,s,rows,w,fixture,_,_,base=prep("held")
    new=Critic(parse(blob))
    with torch.no_grad():
        pred=torch.cat([new(h[i:i+32],s[i:i+32]) for i in range(0,len(h),32)])
    for i,r in enumerate(rows):
        r.update(parent=float(base[i]),new=float(pred[i]),weight=float(w[i]))
    doc=dict(protocol=PROTO,weights_sha256=sha(blob),held_open_after_fit_sha=sha((H/"fit.json").read_bytes()),
        all=grade(rows),domains={k:grade([r for r in rows if r["domain"]==k]) for k in ["h2h","ladder"]},
        phases={k:grade([r for r in rows if a<=r["step"]<b]) for k,a,b in [("opening",0,100),("middle",100,200),("end",200,300)]})
    doc["point_gate"]=all(section["estimates"]["new"]["MAE"] <= section["estimates"]["parent"]["MAE"] for section in [doc["domains"]["h2h"]]) and doc["all"]["estimates"]["new"]["MAE"]<doc["all"]["estimates"]["parent"]["MAE"]
    file(H/"held_rows.json.gz",gzip.compress((json.dumps(rows,allow_nan=False,separators=(",",":"))+"\n").encode(),mtime=0))
    doc["row_sha256"]=sha((H/"held_rows.json.gz").read_bytes())
    dataset=np.stack([x for x,_ in fixture]);sc=np.stack([s for _,s in fixture])
    outs={};checks={}
    for tag,raw in [("parent",parent()),("learned",(H/"candidate.py").read_bytes())]:
        m=module(raw);tn=m._TorchNet(m._POLICY,parse(weight_blob(raw)));wn=m._WinoNet(tn)
        outs[tag]={name:backend(dataset,m._SID,sc) for name,backend in [("Torch",tn),("Wino",wn)]}
        checks[tag+"_value_Torch_vs_Wino_max_abs"]=float(np.abs(outs[tag]["Torch"][2]-outs[tag]["Wino"][2]).max())
        codes=[i for i,r in enumerate(rows) if r["step"] in [6,81,156,291]][:len(fixture)]
        from_cache=[rows[i]["parent" if tag=="parent" else "new"] for i in codes]
        checks[tag+"_cache_vs_Torch_max_abs"]=float(np.max(np.abs(outs[tag]["Torch"][2]-from_cache)))
        assert max(checks[tag+"_value_Torch_vs_Wino_max_abs"],checks[tag+"_cache_vs_Torch_max_abs"])<1e-4
    for backend in ["Torch","Wino"]:
        a,b=outs["parent"][backend],outs["learned"][backend]
        assert all(np.array_equal(a[k],b[k]) for k in (0,1))
        checks[backend+"_policy_worker_and_base_logits_bitwise_equal"]=True
        checks[backend+"_actual_value_max_change"]=float(np.abs(a[2]-b[2]).max())
    assert (H/"candidate.py").read_bytes()==package(blob)
    doc["backend_checks"]={**checks,"input_boards":len(fixture),"single_file_bytes":len(package(blob))}
    save(H/"held.json",doc)
    for k,d in [("all",doc["all"]),*doc["domains"].items()]:
        print(k,d["maps"],d["roots"],d["estimates"])

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["fit","test"])
    args=p.parse_args()
    assert sha((H/"protocol.json").read_bytes())==PROTO
    torch.manual_seed(487113);torch.set_num_threads(1)
    print(fit() if args.mode=="fit" else test(),flush=True)
