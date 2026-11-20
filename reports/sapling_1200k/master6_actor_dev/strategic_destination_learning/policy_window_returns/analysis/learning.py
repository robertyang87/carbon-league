"""Fit, freeze held choices, then score archived original responses in three separate stages."""
import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from runtime import _win_input, _win_option

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
STUDY=EXP.parent
SPEC=HERE/"protocol.json"
PLAN=EXP/"protocol.json"
BC=STUDY/"bc.py"
ELITE=STUDY/"elite_direct_policy/trained.py"
HELD_PRED=HERE/"held_predictions.json"
FIT=HERE/"fit.json"


def check(value,why):
    if not value:
        raise ValueError(why)


def read(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix==".gz" else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    content=(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    immutable(path,content)


def immutable(path,content):
    if path.exists():
        check(path.read_bytes()==content,"Existing frozen artifact differs: "+str(path))
        return
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
        tmp.write(content)
        name=tmp.name
    try:
        os.link(name,path)
    except FileExistsError:
        check(content==path.read_bytes(),"Concurrent artifact mismatch")
    finally:
        os.unlink(name)


def tickets(role):
    plan,rule=read(PLAN),read(SPEC)
    check(sha(EXP/"clearance.json")==rule["clearance_sha256"] and sha(BC)==rule["bc_sha256"] and sha(ELITE)==rule["elite_sha256"],"Changed research input")
    keys=[r for r in plan["schedule"] if r["role"]==role]
    check([r["seed"] for r in keys]==rule["fit_seeds" if role=="fit" else "held_seeds"] and plan["windows"]==rule["window_starts"] and plan["window_length"]==rule["duration"],"Unexpected maps or intervention")
    return keys


def locate(root,ticket,filename):
    direct=root/str(ticket["seed"])/filename
    if direct.is_file():
        return direct
    # Allow nested copied workflow artifact names, never infer a seed from outcome contents.
    candidates=[p for p in root.rglob(filename) if str(ticket["seed"]) in str(p.parent)]
    check(len(candidates)==1,"Required unique artifact "+filename+" for "+str(ticket["seed"]))
    return candidates[0]


def source_only(root,ticket):
    source_path=locate(root,ticket,"source.json.gz")
    src=read(source_path)
    check((src["seed"],src["seat"],src["kind"],src["terrain"],src["protocol"])==
          (ticket["seed"],ticket["seat"],ticket["role"],read(PLAN)["physical_boards"][str(ticket["seed"])],sha(PLAN)),"Source registration")
    check([r["step"] for r in src["rows"]]==list(range(len(src["rows"]))),"Unordered source")
    # Only action-time public inputs enter fitting/prediction. The official match file
    # also has post-action columns, which are never passed to the gate.
    return src,source_path


def root_outcome(root,ticket,start,source_path):
    path=locate(root,ticket,f"root_{start}.json.gz")
    summary=read(locate(root,ticket,"summary.json"))
    record=read(path)
    entry,=[q for q in summary["roots"] if q["step"]==start]
    check(entry["sha256"]==sha(path) and record["source_sha256"]==summary["source_sha256"]==sha(source_path)
          and record["protocol"]==summary["protocol"]==sha(PLAN) and record["seed"]==ticket["seed"],"Unverified branch identity")
    if record["status"]=="unreachable_source_terminal":
        return dict(status="unreachable",raw_margin=0.,raw_win=0.,margin=0.,win=0.,root_sha256=sha(path)),record
    check(record["status"]=="complete","Incomplete intervention")
    bc,elite=[record["arms"][a] for a in ("bc","elite")]
    integrity=record["intervention"]
    state=integrity["shared_prefix"]
    check(state and state[0]["enemy_orders_equal"],"Opponent root command mismatch")
    dirty=any(not q["enemy_orders_equal"] for q in state)
    same=bool(integrity["same_physical_orders_through_window"])
    status="rival_changed_on_common_board" if dirty else "identical_window" if same else "physical_change"
    check(dirty or same or any(not q["own_orders_equal"] and q["enemy_orders_equal"] for q in state),"Unclassified intervention")
    rw,rd=float(elite["terminal"]["win"]-bc["terminal"]["win"]),float(elite["terminal"]["margin"]-bc["terminal"]["margin"])
    check(rw==record["win_delta"] and abs(rd-record["margin_delta"])<1e-5,"Official outcome mismatch")
    return dict(status=status,raw_margin=rd,raw_win=rw,margin=None if dirty else 0. if same else rd,
                win=None if dirty else 0. if same else rw,root_sha256=sha(path),shared_prefix=state,
                officials={a:record["arms"][a]["terminal"] for a in ("bc","elite")}),record


def verify_ticket(folder,plan):
    import sys
    sys.path.insert(0,str(EXP/"runner"))
    import verify
    return verify.verify(folder,plan)


def build(coef,output):
    text=BC.read_text()
    elite_source=ELITE.read_text()
    a,b=text.splitlines(keepends=True),elite_source.splitlines(keepends=True)
    differences=[(x,y) for x,y in zip(a,b,strict=True) if x!=y]
    check(len(differences)==1 and differences[0][0].startswith("_W64 = "),"Policies differ outside Sapling student payload")
    blob=ast.literal_eval(differences[0][1].split(" = ",1)[1])
    addition="\n_WIN_BC_NET=None\n_WIN_ELITE_NET=None\n_WIN_AT=None\n_WIN_COEF="+repr(tuple(map(float,coef)))+"\n_WIN_ELITE_W="+repr(blob)+"\n"
    package=(text+addition+(HERE/"runtime.py").read_text()+"\n_win_install()\n").encode()
    check(len(package)<=10_000_000,"Deployment size over limit")
    immutable(output,package)


def fit(root):
    items=[]; verified=[]
    for ticket in tickets("fit"):
        src,sp=source_only(root,ticket)
        verified.append(verify_ticket(sp.parent,PLAN))
        for start in (256,280):
            res,_=root_outcome(root,ticket,start,sp)
            x=list(_win_input(src["rows"][start]["public"],src["configuration"])) if start<len(src["rows"]) else None
            items.append(dict(**ticket,start=start,public_x=x,source_sha=sha(sp),**res))
    eligible=[i for i in items if i["status"] in ("physical_change","identical_window")]
    maps={i["seed"] for i in eligible}
    changed=[i for i in eligible if i["status"]=="physical_change"]
    supported=len(maps)>=3 and len(changed)>=4
    if supported:
        x=np.asarray([v["public_x"] for v in eligible],np.float64)
        y=np.asarray([v["margin"]/1000 for v in eligible],np.float64)
        counts={s:sum(i["seed"]==s for i in eligible) for s in maps}
        w=np.array([1/(len(maps)*counts[i["seed"]]) for i in eligible])
        a=x.T@(w[:,None]*x)+.5*np.eye(x.shape[1])
        b=x.T@(w*y)
        coef=np.linalg.solve(a,b)
        check(np.isfinite(coef).all(),"Nonfinite gate")
    else:
        coef=np.zeros(4)
    release=HERE/"candidate.py"
    build(coef,release)
    save(FIT,dict(spec=sha(SPEC),plan=sha(PLAN),method=sha(Path(__file__)),runtime=sha(HERE/"runtime.py"),
                  supported=supported,fit_tickets=verified,maps=len(maps),usable=len(eligible),changed=len(changed),
                  coef=coef.tolist(),package=sha(release),bytes=release.stat().st_size,items=items))


def predict(root):
    fit=read(FIT)
    check(fit["spec"]==sha(SPEC) and fit["plan"]==sha(PLAN) and fit["package"]==sha(HERE/"candidate.py"),"Model and plan must be frozen")
    maps=[]
    for ticket in tickets("hold"):
        src,sp=source_only(root,ticket)
        windows=[]
        for start in (256,280):
            if start<len(src["rows"]):
                obs=src["rows"][start]["public"]
                x=_win_input(obs,src["configuration"])
                z=sum(i*j for i,j in zip(x,fit["coef"]))
                windows.append(dict(t=start,reached=True,x=list(x),logit=z,elite=bool(z>0.)))
            else:
                windows.append(dict(t=start,reached=False,x=None,logit=None,elite=False))
        first=next((w["t"] for w in windows if w["elite"]),None)
        maps.append(dict(**ticket,source=sha(sp),windows=windows,first=first))
    save(HELD_PRED,dict(fit=sha(FIT),coef=fit["coef"],package=fit["package"],held=maps))


def module(path):
    spec=importlib.util.spec_from_file_location("tested_window_release",path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def check_entry(src,ticket,map_pred,roots):
    p=module(HERE/"candidate.py")
    seat=ticket["seat"]
    first=map_pred["first"]
    chosen=(roots[first]["arms"]["elite"]["rows"] if first else [])
    sequence=src["rows"] if first is None else src["rows"][:first]+chosen
    modes={"elite":0,"bc":0}
    for line in sequence:
        t=line["step"]
        order=p.agent(line["public"],src["configuration"])
        expected=line["orders"][seat]
        check(order==expected,"Normal entry command differs at "+str((ticket["seed"],t)))
        elite=first is not None and first<=t<first+8
        if t:
            check((p._NETS[1] is p._WIN_ELITE_NET)==elite,"Wrong student selection")
        modes["elite" if elite else "bc"]+=1
    return modes


def estimate(targets,predictions,first,field):
    values=[j[field] for j in targets]
    if None in values:
        return None
    i=None if first is None else (256,280).index(first)
    sequence=values[i] if i is not None else 0.
    choices=[w["elite"] for w in predictions]
    average=float(np.mean([v*x for v,x in zip(values,choices)]))
    return dict(single_vs_bc=sequence,single_vs_elite_256=sequence-values[0],
                single_vs_elite_280=sequence-values[1],both_roots_vs_bc=average,
                both_roots_vs_elite=average-float(np.mean(values)))


def assess(root):
    pred=read(HELD_PRED); fit=read(FIT)
    check(pred["fit"]==sha(FIT) and pred["package"]==fit["package"]==sha(HERE/"candidate.py"),"Frozen prediction identity")
    rows=[]
    for ticket,mp in zip(tickets("hold"),pred["held"],strict=True):
        src,sp=source_only(root,ticket)
        check(sha(sp)==mp["source"] and ticket["seed"]==mp["seed"],"Held allocation changed")
        verified=verify_ticket(sp.parent,PLAN)
        results=[];full={}
        for t in (256,280):
            item,full[t]=root_outcome(root,ticket,t,sp)
            results.append(item)
        modes=check_entry(src,ticket,mp,full)
        contrasts={f:estimate(results,mp["windows"],mp["first"],f) for f in ("raw_margin","raw_win","margin","win")}
        applied=mp["first"]
        moved=bool(applied and results[(256,280).index(applied)]["status"]=="physical_change")
        rows.append(dict(**mp,results=results,verification=verified,public_entry=modes,applied_physical_change=moved,contrasts=contrasts))
    summary={}
    for f in ("raw_margin","raw_win","margin","win"):
        q=[r["contrasts"][f] for r in rows]
        if None in q:
            summary[f]=None
        else:
            summary[f]={k:dict(mean=float(np.mean([v[k] for v in q])),
                positive=sum(v[k]>0 for v in q),leave_one_map_means=[float(np.mean([u[k] for j,u in enumerate(q) if i!=j])) for i in range(len(q))]) for k in q[0]}
    cm,cw=summary["margin"],summary["win"]
    gates=dict(fit=fit["supported"],identifiable=cm is not None and cw is not None,
               margin=cm is not None and all(v["mean"]>0 for v in cm.values()),
               win=cw is not None and all(v["mean"]>=0 for v in cw.values()),
               at_least_one_map=cm is not None and all(v["positive"]>0 for k,v in cm.items() if k.startswith("single_")),
               changed=sum(r["applied_physical_change"] for r in rows)>=2,
               multiple_choices=len({r["first"] for r in rows})>=2)
    save(HERE/"assessment.json",dict(held_predictions=sha(HELD_PRED),maps=rows,summary=summary,gates=gates,
         development_eligible=all(gates.values()),formal_games=0,fresh_games_of_selector=0))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("fit","predict","assess"))
    parser.add_argument("data",type=Path)
    args=parser.parse_args()
    {"fit":fit,"predict":predict,"assess":assess}[args.stage](args.data)
