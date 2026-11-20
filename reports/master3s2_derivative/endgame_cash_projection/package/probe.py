"""Replay the known public roots on the exact release and on a separately identified budget-controlled reconstruction."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
import time

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
sys.path.insert(0,str(D/"opponent_policy"))
sys.path.insert(0,str(HERE.parent))
from original_policy import free_entry, module, PROTOCOL
import runtime
import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix==".gz" else raw)


def require(cond,msg):
    if not cond:raise ValueError(msg)


def controlled(raw):
    """Exact source differences to recover archived menu/depth, never a live timer measurement."""
    src=raw.decode().replace("\r\n","\n")
    for old,new in [
        ("    cands = [act0]; keys","    most = _archived_k\n    cands = [act0]; keys"),
        ("        return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT","        return False"),
        ("_cash_pick(best, alive, mean, dep, run, cands, depth, rolls, cut, step, me, configuration, t0)",
         "_cash_pick(best, alive, mean, dep, run, cands, depth, rolls, cut, step, me, configuration, _time.perf_counter())")]:
        require(src.count(old)==1,"Different recovery anchor")
        src=src.replace(old,new)
    return src.encode()


def restore(model,ticket,card,root):
    tracker=model.BeliefTracker(model._Bel,ticket["seat"]);previous=None
    for t,obs,cfg,issued in free_entry.public_history(card):
        if t<ticket["step"]:
            tracker.step(obs["players"],t,obs["carbon"])
            tracker.set_actions(obs["players"],issued);previous=obs["players"]
        else:
            require(all(root["public_root"][k]==obs[k] for k in ("player","step","players","trees","carbon")),"Visible state mismatch")
            return tracker,previous,dict(obs,remainingOverageTime=root["public_root"]["remainingOverageTime"]),{k:v for k,v in cfg.items() if k!="randomSeed"}
    raise ValueError("Missing root")


def replay(model,ticket,root,state,archived,heads):
    import copy
    tracker,previous,obs,cfg=state
    model._ST=dict(tr=copy.deepcopy(tracker),prev=previous,rng=np.random.RandomState(ticket["policy_rng"]))
    model._S_COST[:]=root["cost_before_root"]
    model._S_REC=[]
    model._archived_k=root["clock"]["K"]
    capture={};meter=[];starts=[]
    m=sys.monitoring;slot=4;code=model._search.__code__
    selector=getattr(model,"_cash_pick",None)
    def start(c,offset):
        starts.append(time.perf_counter())
    def finish(c,offset,retval):
        if selector is not None and c==selector.__code__:
            meter.append(time.perf_counter()-starts.pop())
        elif c==code:
            capture.update(sys._getframe(1).f_locals)
    m.use_tool_id(slot,"capture original-order finalist identifiers")
    try:
        m.register_callback(slot,m.events.PY_START,start)
        m.register_callback(slot,m.events.PY_RETURN,finish)
        m.set_local_events(slot,code,m.events.PY_RETURN)
        if selector:m.set_local_events(slot,selector.__code__,m.events.PY_START|m.events.PY_RETURN)
        began=time.perf_counter()
        order=model.agent(obs,cfg)
        elapsed=time.perf_counter()-began
    finally:
        m.set_local_events(slot,code,0)
        if selector:m.set_local_events(slot,selector.__code__,0)
        m.register_callback(slot,m.events.PY_START,None);m.register_callback(slot,m.events.PY_RETURN,None);m.free_tool_id(slot)
    menu=capture.get("cands",[order]);alive=capture.get("alive",[]);values=capture.get("mean",{})
    pair=capture.get("cash_pair")
    if pair is None and len(values)>=2:
        pair=sorted(alive,key=values.__getitem__,reverse=True)[:2]
    signed=[a["order"] for a in root["arms"]]
    observed=[menu[i] for i in (pair or [])]
    scores=capture.get("cash_scores")
    why=capture.get("cash_reason")
    if why=="compared":
        for idx,score in zip(pair,scores):
            branch=next(b for b in capture["run"] if b[0]==idx and b[1]==0)
            players=branch[2]["players"];s=ticket["seat"]
            given=players[s][0]-players[1-s][0]+runtime.predict_residual(runtime.features(players,s,299-branch[2]["step"]),heads)
            require(abs(score-given)<1e-8,"Packaged features differ from training runtime")
        require(order==observed[int(scores[1]>scores[0])],"Public order is not the projected choice")
    expected=signed[archived["choice"]["enhanced"]]
    return dict(wall_s=elapsed,selector_wall_s=sum(meter),returned_order=order,root_k=len(menu),
                root_depth=capture.get("depth",0),rolls=capture.get("rolls",0),cut=capture.get("cut"),
                surviving_count=len(alive),candidate_indices=pair,identified_pair=observed,
                pair_matches=observed==signed,menu_matches=menu==root["menu"],native_first_matches=bool(observed and observed[0]==root["issued"]),
                returned_expected=order==expected,returned_archived_second=order==signed[1],projection_scores=scores,reason=why)


def main(mode):
    import torch
    os.environ["CARBON_DEBUG"]="1";torch.set_num_interop_threads(1)
    raw=(HERE/"candidate.py").read_bytes();parent=(D/"cold_start/candidate.py").read_bytes()
    reg=load(HERE/"manifest.json")
    require(sha(raw)==reg["sha256"] and sha(parent)==reg["parent_sha256"] and os.sched_getaffinity(0)=={3},"Identity or affinity")
    heads=load(HERE.parent/"weights.json")["enhanced"]
    contract=load(D/"action_rank_data/protocol.json")
    cases={r["ordinal"]:r for r in load(HERE.parent/"paired_inputs.json.gz")["rows"]}
    cards={c["sha256"]:c for c in PROTOCOL["games"]}
    models={"candidate":module(controlled(raw) if mode=="controlled" else raw)}
    if mode=="live":models["parent"]=module(parent)
    torch.set_num_threads(1)
    rows=[]
    for ticket in contract["tickets"]:
        i=ticket["ordinal"];s=ticket["seat"];t=ticket["step"]
        p=D/"action_rank_data"/ticket["split"]/f"{ticket['seed']}_{s}_{t}_root.json.gz"
        r=load(p);a=cases[i]
        require(sha(p.read_bytes())==a["root_sha256"] and r["ticket"]==ticket,"Root identity")
        item=dict(ordinal=i,seat=s,seed=ticket["seed"],step=t,archived_status=a["status"])
        if a["status"]=="paired":
            state=restore(models["candidate"],ticket,cards[ticket["source_sha256"]],r)
            item["archived_prefers_second"]=bool(a["choice"]["enhanced"])
            items=list(models.items())
            if i%2:items=items[::-1]
            for k,v in items:
                item[k]=replay(v,ticket,r,state,a,heads)
            print(mode,i,item["candidate"]["root_depth"],item["candidate"]["reason"],item["candidate"]["pair_matches"],flush=True)
        else:
            item["reason"]=a["reason"]
        rows.append(item)
    trials=[r for r in rows if "candidate" in r]
    results=[r["candidate"] for r in trials]
    summary=dict(eligible_archived=len(trials),verified_menus=sum(r["menu_matches"] for r in results),
                 identical_archived_pairs=sum(r["pair_matches"] for r in results),compared=sum(r["reason"]=="compared" for r in results),
                 matched_expected_command=sum(r["returned_expected"] for r in results),
                 projected_archived_switches=[r["ordinal"] for r in trials if r["archived_prefers_second"] and r["candidate"]["reason"]=="compared" and r["candidate"]["pair_matches"] and r["candidate"]["returned_expected"]],
                 original_expected_switches=[r["ordinal"] for r in trials if r["archived_prefers_second"]],
                 max_selector_ms=max(r["selector_wall_s"] for r in results)*1000,median_selector_ms=median(r["selector_wall_s"] for r in results)*1000,
                 max_candidate_call_ms=max(r["wall_s"] for r in results)*1000)
    output=dict(mode=mode,release_sha256=sha(raw),parent_sha256=sha(parent),weights_sha256=reg["weights_sha256"],
                executed_sha256=sha(controlled(raw)) if mode=="controlled" else sha(raw),affinity=sorted(os.sched_getaffinity(0)),
                cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),summary=summary,rows=rows)
    dst=HERE/(mode+".json");require(not dst.exists(),"Prior trial exists")
    dst.write_text(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(mode,summary,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("mode",choices=("live","controlled"));main(ap.parse_args().mode)
