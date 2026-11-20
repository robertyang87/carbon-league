"""Audit the identified plant/leave contrast or collect one native full-response pair."""
import copy
import gzip
import itertools
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
sys.path.insert(0, str(D/"action_rank_data"))
import collect as native
from history import advance, freight, setup, sha, token, views
from branches import continue_pair, restore_prefix

tree = freight.load("tail_public_geometry", D/"ladder/tree_payback/runtime.py")
REG = json.loads((HERE/"protocol.json").read_bytes())
P = json.loads((D/"action_rank_data/protocol.json").read_bytes())
check = freight.check
check(sha((HERE/"protocol.json").read_bytes())=="65c63e4a8c75e8a1cd0105f3009402eaa40469a8eac56c61b269e5772767c3f0", "Changed prospective design")
check(sha((D/"action_rank_data/protocol.json").read_bytes())==REG["historical_pair_registry"] and
      P["base"]==REG["parent"] and P["examiner"]==REG["examiner"], "Changed source exam")
DIRS = ("UP","RIGHT","DOWN","LEFT")


def dump(name, value):
    return native.save_json(HERE/name, value)


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix==".gz" else path.read_bytes())


def empty_planters(obs, cfg):
    ps,s=obs["players"],obs["player"]
    excluded={z[0] for p in ps for z in p[3].values()}|{v for p in ps for v in p[1].values()}
    if ps[s][0]<tree.quoted_price(ps,s,cfg):
        return {}
    return {k:w[0] for k,w in ps[s][2].items() if w[2]=="PLANTER" and w[0] not in excluded}


def contrasts(obs, cfg, a, b):
    planters=empty_planters(obs,cfg)
    diffs=sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))
    flags=[]
    for k in diffs:
        if k not in planters or ((k in a)==(k in b)):
            continue
        action=(a.get(k) or b.get(k))
        if action not in DIRS:
            continue
        s=obs["player"];own=obs["players"][s];cell=planters[k]
        def dest(pos,order):
            return tree.neighbors(pos,cfg["size"])[DIRS.index(order)] if order else pos
        plant_first=k not in a
        collisions=[]
        for command in (a,b):
            target=dest(cell,command.get(k))
            reserved={dest(w[0],command.get(j)) for j,w in own[2].items() if j!=k}
            reserved.update(p for u,p in own[1].items() if command.get(u))
            collisions.append(target in reserved)
        flags.append(dict(worker=k,cell=cell,direction=action,plant_first=plant_first,
                          cochanged=sorted(set(diffs)-{k}),friendly_conflict=collisions))
    return flags


def unique_pair(obs,cfg,a,b):
    return next((x for x in contrasts(obs,cfg,a,b) if not x["cochanged"] and not any(x["friendly_conflict"])),None)


def audit():
    rows=[]
    for t in P["tickets"]:
        stem=f"{t['seed']}_{t['seat']}_{t['step']}"
        folder=D/"action_rank_data"/t["split"]
        root=read(folder/(stem+"_root.json.gz"));pair=read(folder/(stem+"_pair.json"))
        row=dict(ordinal=t["ordinal"],seed=t["seed"],terrain=t["terrain"],step=t["step"],seat=t["seat"],split=t["split"],
                 H=root["depth"],status=pair["status"])
        if pair["status"]=="complete":
            obs=root["public_root"];cfg=freight.econ.config()
            a,b=root["arms"]
            fa=contrasts(obs,cfg,a["order"],b["order"])
            joint=unique_pair(obs,cfg,a["order"],b["order"])
            eligible=[int(k) for k in root["all_final_values"]]
            possible=[(i,j,unique_pair(obs,cfg,root["menu"][i],root["menu"][j])) for i,j in itertools.combinations(eligible,2)]
            row.update(any_fresh_plant_contrast=fa,strict=joint,other_native_finalist_pairs=[[i,j,r] for i,j,r in possible if r],
                       native_first_minus_second=a["original_value"]-b["original_value"],
                       official_margin_first_minus_second=pair["arms"]["first"]["margin"]-pair["arms"]["second"]["margin"])
            if joint:
                tape,acts,_,env,sim=setup(t,source(t))
                for actions in acts[:t["step"]]:advance(env,sim,actions)
                debit=[]
                for arm in ("first","second"):
                    trace=read(folder/(stem+"_"+arm+".json.gz"))
                    mirror=copy.deepcopy(sim);mirror.cash.events.clear(); mirror.step(*trace["rows"][0]["orders"])
                    actual=[ev[2] for ev in mirror.cash.events if ev[0]=="plant" and ev[1]==t["seat"] and ev[3]==joint["worker"] and ev[4]==joint["cell"]]
                    debit.append(actual)
                row["actual_root_debits"]=debit
                row["both_interventions_verified"]=bool(debit[0])==joint["plant_first"] and bool(debit[1])!=joint["plant_first"]
        rows.append(row)
    out={"rows":rows,"sufficient_for_residual":all(sum(bool(r.get("both_interventions_verified")) for r in rows if r["split"]==split)>=n for split,n in (("train",4),("held",2)))}
    dump("archived_pair_audit.json",out)
    return out


def source(ticket):
    return next(x for x in read(D/"opponent_policy/protocol.json")["games"] if x["sha256"]==ticket["source_sha256"])


def monitor(mod,obs,cfg):
    trace={}
    def capture(code,offset,result):
        trace.update(sys._getframe(1).f_locals)
    mon=sys.monitoring; tool=4
    mon.use_tool_id(tool,"focal tree finalist selection")
    try:
        mon.register_callback(tool,mon.events.PY_RETURN,capture)
        mon.set_local_events(tool,mod._search.__code__,mon.events.PY_RETURN)
        own=freight.measured(mod,obs,cfg)
    finally:
        mon.set_local_events(tool,mod._search.__code__,0)
        mon.register_callback(tool,mon.events.PY_RETURN,None)
        mon.free_tool_id(tool)
    menu=trace.get("cands",[own[0]])
    ix=trace.get("alive",[]);v=trace.get("val",{});dep=trace.get("dep",{})
    finalists={i:float(v[i,0]) for i in ix if dep.get(i)==8 and (i,0) in v}
    good=trace.get("depth")==8 and not trace.get("cut",True) and trace.get("rolls")==1 and len(finalists)==len(ix) and len(ix)>=2
    opts=[]
    rank=sorted(finalists,key=lambda i:-finalists[i])
    if good:
        for i,j in itertools.combinations(rank,2):
            info=unique_pair(obs,cfg,menu[i],menu[j])
            if info:
                opts.append((int(own[0] not in (menu[i],menu[j])),rank.index(i),rank.index(j),i,j,info))
    result=dict(own=own[0],clock=own[1],native_scores=finalists,menu=menu,complete=good)
    if opts:
        _,_,_,i,j,info=min(opts,key=lambda x:x[:3])
        x,y=(i,j) if info["plant_first"] else (j,i)
        result["pair"]=dict(plant=menu[x],leave=menu[y],plant_index=x,leave_index=y,info=info,
                            plant_value=finalists[x],leave_value=finalists[y])
    return own,result


def pilot():
    control=read(HERE/"archived_pair_audit.json")
    if control["sufficient_for_residual"]:raise ValueError("Have enough registered labels already, do not spend on fallback")
    if (HERE/"selection.json").exists():
        raise RuntimeError("Already committed a selection; do not rerun this sample")
    scans=[]
    for t in [v for v in P["tickets"] if v["split"]=="train"]:
        card=source(t);tape,acts,cfg,env,sim=setup(t,card)
        prospective=[]
        for i in range(270):
            if i in range(249,270,2):
                obs=views(env); affordable=empty_planters(obs[t["seat"]],cfg)
                expected=dict(incoming_cash=list(sim.cash),public_sha256_by_seat=[freight.digest(v) for v in obs],
                              prior_orders_sha256=token(acts[:i]))
                preview=dict(step=i,preaction_ids=affordable,expected=expected)
                prospective.append(preview)
            if i<269:
                advance(env,sim,acts[i])
        for p in prospective:
            row=dict(seed=t["seed"],seat=t["seat"],step=p["step"],terrain=t["terrain"],available=p["preaction_ids"])
            scans.append(row)
            if not row["available"]:
                continue
            ticket=dict(t,step=p["step"])
            state=restore_prefix(ticket,card,p["expected"])
            mods,cfg,env,sim,obs,_=state;s=t["seat"]
            m=mods[s];m._ST["rng"]=freight.np.random.RandomState(t["policy_rng"])
            prefix=freight.original_state.h([v._ST for v in mods])
            rng=freight.original_state.h(m._ST["rng"]);cost=list(m._S_COST)
            own,search=monitor(m,obs[s],cfg); row.update(search)
            print(t["seed"],p["step"],search["clock"]["H"],len(search["menu"]),bool(search.get("pair")),flush=True)
            if "pair" not in search:
                continue
            info=search["pair"]
            admission={k:info[k] for k in ("plant","leave")}
            admission.update(menu_sha256=token(search["menu"]),cost=cost)
            dump("selection.json",dict(protocol_sha256=sha((HERE/"protocol.json").read_bytes()),
                 status="selected_before_original_reply",ticket=ticket,intervention=info,
                 scans=scans,remaining="budget-censored; first qualified root"))
            r=continue_pair(ticket,admission,["plant","leave"],state,own,prefix,rng,dump)
            dump("response.json",r)
            return
    dump("selection.json",dict(status="none",scans=scans))

if __name__=="__main__":
    if sys.argv[1]=="audit":audit()
    elif sys.argv[1]=="pilot":
        check(len(os.sched_getaffinity(0))==1,"Pin the real search to a single logical CPU")
        pilot()
    else:raise ValueError("Use audit or pilot")
