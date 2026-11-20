"""Reconstruct eight registered root interventions and fork the two policies only when orders differ."""
import ast
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
sys.path.insert(0,str(D/"symmetry_value/terminal_pairs"))
from branches import continue_pair, freight

REG=json.loads((HERE/"protocol.json").read_bytes())
PATHS={"parent":D/"cold_start/candidate.py","head":HERE.parent/"package/candidate.py",
       "original":D.parents[1]/"data/master3s2_source.py"}
BUDGET=300.0


def h(x):
    return hashlib.sha256(x).hexdigest()


def canonical(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()


def save(path,content):
    raw=canonical(content)+b"\n"
    if path.suffix==".gz":
        raw=gzip.compress(raw,mtime=0)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".partial")
    tmp.write_bytes(raw)
    os.replace(tmp,path)
    return h(raw)


def need(test,text):
    if not test:
        raise ValueError(text)


def control_fn(model,source):
    text=source.decode().replace("\r\n","\n")
    tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_search")
    code="\n".join(text.splitlines()[node.lineno-1:node.end_lineno])
    a="    cands = [act0]; keys"
    b="        return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT"
    need(code.count(a)==code.count(b)==1,"unrecognized budget controller")
    code=code.replace(a,"    most = _response_cap\n"+a)
    # Simulate only the completed depth count of the unmodified parent on this machine.
    code=code.replace(b,"        return t > _response_depth")
    if "_cash_pick(" in code:
        a="depth, rolls, cut, step, me, configuration, t0)"
        need(code.count(a)==1,"Unrecognized selector clock")
        code=code.replace(a,"depth, rolls, cut, step, me, configuration, _time.perf_counter())")
    namespace={}
    exec(compile(code,"<equal-work-public-search>","exec"),model.__dict__,namespace)
    return namespace["_search"]


def snap(mod):
    return copy.deepcopy(freight.original_state.dynamic(mod)),copy.deepcopy(mod._RX_CFG[1])


def restore(mod,state,cfg):
    values,parsed=copy.deepcopy(state)
    for k,v in values.items():
        setattr(mod,k,v)
    mod._RX_CFG[:]=[cfg if parsed else None,parsed]


def call(mod,ob,cfg):
    captured={}
    code=mod._search.__code__
    mon=sys.monitoring
    def grab(_,offset,returned):
        f=sys._getframe(1).f_locals
        menu=f.get("cands",[returned])
        captured.update(menu=menu, native_mean={str(k):float(x) for k,x in f.get("mean",{}).items()},
                        dep={str(k):v for k,v in f.get("dep",{}).items()},
                        cash_reason=f.get("cash_reason"),cash_scores=f.get("cash_scores"),pair=f.get("cash_pair"))
    slot=4
    mon.use_tool_id(slot,"public search output and native depth")
    try:
        mon.register_callback(slot,mon.events.PY_RETURN,grab)
        mon.set_local_events(slot,code,mon.events.PY_RETURN)
        command,clock=freight.measured(mod,ob,cfg)
    finally:
        mon.set_local_events(slot,code,0);mon.register_callback(slot,mon.events.PY_RETURN,None);mon.free_tool_id(slot)
    if not captured:
        raise ValueError("Public search frame missing")
    return dict(order=command,clock=clock,**captured)


def history_state(mod,state,seat,rng):
    prefix=state["prefix"]
    need(len(prefix)==state["public"][seat]["step"],"Source prefix length")
    cfg=state["cfg"]
    mod._ST={};mod._S_COST[:]=[0.0]*4
    mod._S_STATS=copy.deepcopy(fresh_stats)
    for i,(obs,actions) in enumerate(prefix):
        visible=obs[seat];p=visible["players"]
        if i<3:
            freight.measured(mod,visible,cfg)
        else:
            mod._ST["tr"].step(p,i,visible["carbon"])
        mod._ST["tr"].set_actions(p,actions[seat]);mod._ST["prev"]=p
    # Historic RNG was naturally drawn and is not archived. The experimental root seed is registered.
    mod._ST["rng"]=np.random.RandomState(rng)


def compute(ticket,mods,controlled,start_budget):
    from recovery import source
    state=source(ticket)
    s,t,seed=ticket["seat"],ticket["step"],ticket["policy_rng"]
    cfg=state["cfg"];pub=state["public"]
    parent,head,op=(mods[k] for k in ("parent","head","original"))
    history_state(parent,state,s,seed)
    history_state(op,state,1-s,seed^0x7d4a1f6b)
    before=snap(parent)
    restore(head,before,cfg)
    live={}
    order=["parent","head"] if ticket["ordinal"]%2==0 else ["head","parent"]
    for tag in order:
        live[tag]=call(mods[tag],pub[s],cfg)
    post_live=snap(parent)
    k,d,rolls=(live["parent"]["clock"][n] for n in ("K","H","rolls"))
    work={}
    for tag in order[::-1]:
        m=mods[tag];restore(m,before,cfg)
        original=m._search
        try:
            m._response_cap=k;m._response_depth=d;m._search=controlled[tag]
            work[tag]=call(m,pub[s],cfg)
        finally:
            m._search=original
    equal=work["parent"]["order"]==work["head"]["order"]
    verify={
        "same_native_menu":work["parent"]["menu"]==work["head"]["menu"]==live["parent"]["menu"],
        "same_native_mean":work["parent"]["native_mean"]==work["head"]["native_mean"]==live["parent"]["native_mean"],
        "same_native_depths":work["parent"]["dep"]==work["head"]["dep"]==live["parent"]["dep"],
        "parent_restored":work["parent"]["order"]==live["parent"]["order"],
        "same_target_work":all(tuple(work[a]["clock"][n] for n in ("K","H","rolls"))==(k,d,rolls) for a in order),
    }
    result=dict(ticket=ticket,source=state["expected"],source_work=state["source_work"],live=live,matched=work,verify=verify,
                unchanged=equal,live_orders_agree_with_matched={a:work[a]["order"]==live[a]["order"] for a in order},
                used_continuation_cpu=0.,delta=None,branch=None)
    if not all(verify.values()):
        result.update(status="unrecoverable_matched_work")
        return result
    if equal:
        result.update(status="equal_commands",delta=0.)
        return result
    reserved=4*(299-t)*.70
    if reserved>start_budget:
        result.update(status="remaining_cpu_insufficient",required_seconds=reserved)
        return result
    native=[op,op];native[s]=parent
    restore(parent,post_live,cfg)
    admission={tag:work[tag]["order"] for tag in order}
    admission["order_source"]="terminal_cash_selector_matched_to_live_parent_work"
    system=(native,cfg,state["env"],state["sim"],pub,state["expected"])
    def file(name,value):
        return save(HERE/"branches"/ticket["arm"]/name,value)
    started=time.process_time()
    def cutoff(*_):
        raise TimeoutError("Preregistered total continuation CPU limit")
    old=signal.signal(signal.SIGPROF,cutoff)
    signal.setitimer(signal.ITIMER_PROF,start_budget)
    try:
        branch=continue_pair(ticket,admission,order,system,(admission["parent"],live["parent"]["clock"]),
                             freight.original_state.h(before[0]),freight.original_state.h(before[0]["_ST"]["rng"]),file)
        a=branch["arms"];result.update(status="scored_pair",delta=a["head"]["margin"]-a["parent"]["margin"],branch=branch)
    except TimeoutError as exc:
        result.update(status="unfinished_at_cpu_cap",error=str(exc))
    except Exception as exc:
        result.update(status="continuation_failure",error=repr(exc))
    finally:
        signal.setitimer(signal.ITIMER_PROF,0);signal.signal(signal.SIGPROF,old)
        result["used_continuation_cpu"]=time.process_time()-started
    return result


def main():
    global fresh_stats
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    need(len(os.sched_getaffinity(0))==1,"Run pinned to a single CPU")
    os.environ["CARBON_DEBUG"]="1"
    mods={};controls={}
    for tag,p in PATHS.items():
        raw=p.read_bytes();need(h(raw)==REG[tag],"Changed actual "+tag)
        mod=freight.load("live_cash_check_"+tag,p);mods[tag]=mod
        if tag!="original":controls[tag]=control_fn(mod,raw)
        need(mod._SEED is None and mod._TEMP==1 and mod._S_ABORT==.58,"Changed native policy resources")
    fresh_stats=copy.deepcopy(mods["parent"]._S_STATS)
    rows=[];left=BUDGET
    for ticket in REG["tickets"]:
        path=HERE/"roots"/(str(ticket["ordinal"])+".json")
        if path.exists():
            result=json.loads(path.read_bytes());need(result["ticket"]==ticket,"Resume mismatch")
        else:
            try:
                result=compute(ticket,mods,controls,left)
            except Exception as exc:
                if isinstance(exc,FileNotFoundError):raise
                import traceback
                result=dict(ticket=ticket,status="failed_recovery",error=repr(exc),trace=traceback.format_exc(),
                            delta=None,used_continuation_cpu=0.)
            save(path,result)
        rows.append(result);left=max(0.,left-result["used_continuation_cpu"])
        print(ticket["ordinal"],result['status'],result['delta'],result.get('error',''),flush=True)
    summary=[{k:v.get(k) for k in ("ticket","status","delta","used_continuation_cpu","live_orders_agree_with_matched")} for v in rows]
    maps={}
    for seed in sorted({r["ticket"]["seed"] for r in rows}):
        values=[r["delta"] for r in rows if r["ticket"]["seed"]==seed]
        maps[seed]=dict(known=sum(x is not None for x in values),values=values,
                        mean=sum(values)/len(values) if all(d is not None for d in values) else None)
    save(HERE/"results.json",dict(roots=summary,by_map=maps,remaining_continuation_cpu=left,protocol_sha256=h((HERE/"protocol.json").read_bytes())))


if __name__=="__main__":
    main()
