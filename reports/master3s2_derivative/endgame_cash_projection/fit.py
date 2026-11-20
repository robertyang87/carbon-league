"""Train only on signed residual official margin and grade predictions by physical map."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import data
import runtime

HERE = Path(__file__).resolve().parent
MAPS_SHA = "30fb6a9b1e3540c6c49410be10005779d154b7b6f1db700b861d78f39ce22a45"
HEADS = ("cash_only","aggregate","enhanced")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path,obj):
    raw=(json.dumps(obj,allow_nan=False,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n").encode()
    path.write_bytes(raw)
    return digest(raw)


def extract(tickets):
    rows,absence=[],[]
    for ticket in tickets:
        game=data.replay(ticket)
        items,reason=data.examples(game)
        source_steps=[]
        config=game["configuration"]
        for item in items:
            t,seat,p = item["step"],item["seat"],item["players"]
            left=config["episodeSteps"]-1-t
            if len(p)!=2 or not np.isfinite(item["final_margin"]):
                raise ValueError("Incomplete or invalid official final score")
            x=runtime.features(p,seat,left,config["size"],config["treeLifespan"])
            anti=runtime.features(p,1-seat,left,config["size"],config["treeLifespan"])
            if any(a != -b for a,b in zip(x,anti)):
                raise ValueError("Antisymmetry violated")
            rows.append(dict(game=ticket["id"],terrain=ticket["terrain"],step=t,seat=seat,
                             remaining=left,cash=p[seat][0]-p[1-seat][0],official=item["final_margin"],x=x))
            source_steps.append(t)
        missing=sorted(set(data.TIMES).difference(source_steps))
        if missing:
            absence.append(dict(game=ticket["id"],terrain=ticket["terrain"],steps=missing,
                                reason="game ended before these registered roots" if reason=="ok" else reason))
    return rows,absence


def learn(rows,names):
    n=len(names)
    counts=Counter(r["terrain"] for r in rows)
    q=np.array([1.0/len(counts)/counts[r["terrain"]] for r in rows])
    x=np.array([r['x'][:n] for r in rows])
    y=np.array([(r["official"]-r["cash"])/1000 for r in rows])
    scale=np.maximum(0.05,np.sqrt(q@(x*x)))
    z=x/scale
    def loss(w):
        d=z@w-y
        h=np.abs(d)
        value=q@np.where(h<=1,0.5*d*d,h-.5)+.01*np.dot(w,w)
        gradient=z.T@(q*np.clip(d,-1,1))+.02*w
        return value,gradient
    result=minimize(loss,np.zeros(n),jac=True,method="L-BFGS-B",
                    options={"maxiter":300,"ftol":1e-13,"gtol":1e-9})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(result.message)
    return dict(features=list(names),scale=scale.tolist(),weights=result.x.tolist(),
                huber_delta=1.0,l2=.02,intercept=0.0,iterations=int(result.nit))


def metrics(group):
    if not group:
        return None
    ans={}
    for name in HEADS:
        errors=np.asarray([r["pred"][name]-r["official"] for r in group])
        ans[name]=dict(mae=float(np.abs(errors).mean()),median_absolute=float(np.median(np.abs(errors))),
                       rmse=float(np.sqrt(np.mean(errors**2))),mean_error=float(np.mean(errors)))
    return ans


def evaluate(rows,heads):
    for r in rows:
        x=r.pop("x")
        p={"cash_only":r["cash"]}
        for name in HEADS[1:]:
            p[name]=r["cash"]+runtime.predict_residual(x,heads[name])
        r['pred']=p
    grouped={k:[r for r in rows if r["game"]==k] for k in sorted({r["game"] for r in rows})}
    result={k:metrics(v) for k,v in grouped.items()}
    map_mae={name:float(np.mean([m[name]["mae"] for m in result.values()])) for name in HEADS}
    comparison={baseline:sum(v["enhanced"]["mae"]<v[baseline]["mae"] for v in result.values())
                for baseline in HEADS[:2]}
    return dict(
        usable_maps=len(grouped),roots=len(rows),by_game=result,by_step={str(t):metrics([r for r in rows if r["step"]==t]) for t in data.TIMES},
        map_equal_mae=map_mae,all_root_metrics=metrics(rows),strictly_improved_maps=comparison,rows=rows,
        pass_primary=all(map_mae["enhanced"]<map_mae[k] for k in HEADS[:2]),
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["train","held"])
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--frozen-weight-sha",default=None,help="Required for held assessment")
    args=p.parse_args()
    manifest=(HERE/"maps.json").read_bytes()
    if digest(manifest)!=MAPS_SHA:
        raise ValueError("Wrong map registration")
    args.out.mkdir(parents=True,exist_ok=True)
    wpath=args.out/"weights.json"
    if args.mode=="held":
        if not args.frozen_weight_sha or digest(wpath.read_bytes())!=args.frozen_weight_sha:
            raise ValueError("Held scoring requires the earlier frozen weights")
    tickets=[t for t in json.loads(manifest)["tickets"] if t["split"]==args.mode]
    if len(tickets)!={"train":160,"held":40}[args.mode]:
        raise ValueError("Changed registered game count")
    rows,absent=extract(tickets)
    if not rows:
        raise ValueError("No valid roots")
    if args.mode=="train":
        heads={"aggregate":learn(rows,runtime.BASE),"enhanced":learn(rows,runtime.FULL),
               "maps_sha256":MAPS_SHA,"runtime_sha256":digest((HERE/"runtime.py").read_bytes()),
               "formula":"final margin = visible cash difference + 1000 times no-intercept standardized linear output"}
        ws=save(wpath,heads)
        print("WEIGHTS",ws,flush=True)
    else:
        heads=json.loads(wpath.read_bytes())
        ws=args.frozen_weight_sha
    if heads["runtime_sha256"]!=digest((HERE/"runtime.py").read_bytes()):
        raise ValueError("Changed deployment feature extraction")
    report=evaluate(rows,heads)
    report.update(split=args.mode,maps_sha256=MAPS_SHA,registered_maps=len(tickets),
                  weights_sha256=ws,unavailable=absent,verdict="DESCRIPTIVE" if args.mode=="train" else
                  "PASS" if report["pass_primary"] else "FAIL")
    target=args.out/("fit.json" if args.mode=="train" else "held.json")
    print("SCORE",save(target,report),report["map_equal_mae"],report["usable_maps"],report["roots"],flush=True)


if __name__=="__main__":
    main()
