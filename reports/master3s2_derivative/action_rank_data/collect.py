"""Extract the two deepest native choices, then branch once against the real original opponent."""
from collections import Counter
import argparse
import gzip
import io
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"symmetry_value/terminal_pairs"))
import branches
from history import advance, freight, setup, sha, token, views
import numpy as np

PLAN_BYTES=(HERE/"protocol.json").read_bytes()
PLAN=json.loads(PLAN_BYTES)
V2=PLAN["base"]


def publish(path,raw):
    freight.check(not path.exists() or path.read_bytes()==raw,"Conflicting stored receipt "+str(path))
    if not path.exists():
        temp=path.with_suffix(".partial")
        temp.write_bytes(raw)
        temp.replace(path)
    return sha(raw)


def save_json(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    return publish(path,gzip.compress(raw,mtime=0) if path.suffix==".gz" else raw)


def rank_call(mod,obs,cfg):
    """Take only the returned _search frame of an unchanged public call; serialize after its clock."""
    trace={}
    def capture(code,offset,result):
        trace.update(sys._getframe(1).f_locals)
    mon=sys.monitoring
    tool=4
    mon.use_tool_id(tool,"native final leaf provenance")
    try:
        mon.register_callback(tool,mon.events.PY_RETURN,capture)
        mon.set_local_events(tool,mod._search.__code__,mon.events.PY_RETURN)
        issued,clock=freight.measured(mod,obs,cfg)
    finally:
        mon.set_local_events(tool,mod._search.__code__,0)
        mon.register_callback(tool,mon.events.PY_RETURN,None)
        mon.free_tool_id(tool)
    menu=trace.get("cands",[issued])
    depth=int(trace.get("depth",0));rolls=int(trace.get("rolls",0))
    detail=dict(menu=menu,issued=issued,clock=clock,depth=depth,rolls=rolls,arms=[])
    if depth!=8 or trace.get("cut",True) or rolls!=1:
        return (issued,clock),dict(**detail,status="skipped",reason="incomplete_native_depth_or_rolls"),{}
    alive=trace["alive"]
    candidates=[i for i in alive if trace["dep"].get(i)==8 and (i,0) in trace["val"]]
    freight.check(len(candidates)==len(alive),"Not all surviving candidates completed the same depth")
    ordered=sorted(candidates,key=lambda i:-trace["val"][i,0])
    ranked=[]
    for i in ordered:
        if all(menu[i]!=menu[prev] for prev in ranked):ranked.append(i)
    if len(ranked)<2:
        return (issued,clock),dict(**detail,status="skipped",reason="less_than_two_distinct_native_orders"),{}
    indices=ranked[:2]
    boards={(b[0],b[1]):b for b in trace["run"]}
    real_inputs={(b[0],b[1]):policy[2] for b,policy in zip(trace["live"],trace["pol"],strict=True)}
    features=[];scalars=[];critic=[];steps=[];terminal=[]
    native_scores={str(i):float(trace["val"][i,0]) for i in alive}
    detail["all_final_values"]=native_scores
    for name,i in zip(("first","second"),indices,strict=True):
        b=boards[i,0]
        state=b[2];me=obs["player"];exact=state["step"]>mod.LAST_TURN
        value=float(trace["val"][i,0]);players=state["players"]
        actual=real_inputs.get((i,0))
        if exact:
            margin=players[me][0]-players[1-me][0]
            freight.check(actual is None and abs(value-mod._terminal(margin))<1e-7,"Wrong exact terminal")
            x=np.zeros((mod.NPL,mod.SIZE,mod.SIZE),np.float32)
        else:
            x=np.asarray(actual,dtype=np.float32)
            rebuilt=trace["view"](b,me)
            freight.check(np.array_equal(rebuilt,x) and x.shape==(mod.NPL,15,15),"Not the native value input")
        u8=np.rint(x*100).astype(np.uint8)
        freight.check(np.array_equal(u8.astype(np.float32)*np.float32(.01),x),"Quantized X round-trip changed")
        scalar=mod._scalars(players,state["step"],me).astype(np.float32)
        features.append(u8);scalars.append(scalar);critic.append(value);steps.append(state["step"]);terminal.append(exact)
        leaf=dict(roll_index=0,leaf_step=state["step"],exact_terminal=exact,players=players,x_float32_sha256=sha(x.tobytes()))
        detail["arms"].append(dict(name=name,candidate_index=i,order=menu[i],original_value=value,leaves=[leaf]))
    arrays=dict(X=np.stack(features)[:,None],scalars=np.stack(scalars)[:,None],critic=np.array(critic,np.float64)[:,None],
                candidate_index=np.array(indices,np.int16),roll_index=np.array([0],np.int16),
                leaf_step=np.array(steps,np.int16)[:,None],depth=np.full((2,1),depth,np.int16),
                exact_terminal=np.array(terminal,np.bool_)[:,None])
    return (issued,clock),dict(**detail,status="paired_eligible"),arrays


def verify_source(ticket):
    source=json.loads((HERE.parent/"opponent_policy/protocol.json").read_bytes())
    card=next(r for r in source["games"] if r["sha256"]==ticket["source_sha256"])
    freight.check(card["expert_seat"]==ticket["seat"] and card["split"]==("train" if ticket["split"]=="train" else "val"),"Wrong split")
    tape,acts,cfg,env,sim=setup(ticket,card);cash=[];expected=None
    for i in range(len(acts)+1):
        cash.append(list(sim.cash))
        if i==ticket["step"]:
            expected=dict(public_sha256_by_seat=[freight.digest(v) for v in views(env)],
                          incoming_cash=list(sim.cash),prior_orders_sha256=token(acts[:i]))
        if i<len(acts):advance(env,sim,acts[i])
    freight.check(token(cash)==card["official_cash_path_sha256"] and [s.reward for s in env.states]==tape["official_score_by_seat"],
                  "Unverified historical source")
    return card,expected


def blind_forecast(ticket, root_sha, leaf_sha=None):
    frozen=json.loads((HERE/"held_training_freeze.json").read_bytes())
    prediction=HERE.parent/"terminal_value_pairs"/f"held_forecast_{ticket['ordinal']}.json"
    # The training-side process writes this immutable pre-response prediction.
    # Waiting here ensures neither branch has consumed the original's root reply.
    while not prediction.exists():
        time.sleep(.5)
    raw=prediction.read_bytes()
    data=json.loads(raw)
    freight.check(data["heads_sha256"]==frozen["frozen_sha256"] and data["root_sha256"]==root_sha
                  and data["terrain"]==ticket["terrain"] and data["seed"]==ticket["seed"]
                  and data["seat"]==ticket["seat"] and data["step"]==ticket["step"],
                  "Blind forecast does not bind the frozen weights and root")
    if leaf_sha is not None:
        freight.check(data["leaf_sha256"]==leaf_sha and np.isfinite(list(data["preds"].values())).all(),
                      "Invalid blind predicted order or leaf identity")
    else:
        freight.check(data["excluded"],"Excluded root lacks a blind receipt")
    return sha(raw)


def collect(ticket):
    folder=HERE/ticket["split"];folder.mkdir(exist_ok=True)
    base=f"{ticket['seed']}_{ticket['seat']}_{ticket['step']}"
    pair_path=folder/(base+"_pair.json")
    if pair_path.exists():
        result=json.loads(pair_path.read_bytes())
        freight.check(result["ticket"]==ticket and result["protocol_sha256"]==sha(PLAN_BYTES),"Conflicting pre-existing pair")
        return result
    if (folder/(base+"_root.json.gz")).exists():
        raise RuntimeError("Unfinished acquisition; preserve original artifacts and finish from checkpoint")
    began=time.perf_counter();cpu=time.process_time()
    card,expected=verify_source(ticket);fees=[Counter(),Counter()]
    restored=branches.restore_prefix(ticket,card,expected,fees)
    mods,cfg,env,sim,observations,_=restored;s=ticket["seat"];mod=mods[s]
    mod._ST["rng"]=np.random.RandomState(ticket["policy_rng"])
    prefix=freight.original_state.h([m._ST for m in mods]);rng=freight.original_state.h(mod._ST["rng"])
    warm_wall=time.perf_counter()-began
    before_cost=list(mod._S_COST)
    own,root,arrays=rank_call(mod,observations[s],cfg)
    roster=observations[s]["players"]
    root.update(ticket=ticket,protocol_sha256=sha(PLAN_BYTES),public_root=observations[s],official_cash=list(sim.cash),
                carried_by_seat=[sum(v[1] for v in p[2].values() if v[2]=="COLLECTOR") for p in roster],
                trees_by_seat=[len(p[3]) for p in roster],prior_registered_economy_by_seat=[dict(v) for v in fees],
                cost_before_root=before_cost,preparation_wall=warm_wall,authority=expected)
    rootsha=save_json(folder/(base+"_root.json.gz"),root)
    metadata=dict(ticket=ticket,protocol_sha256=sha(PLAN_BYTES),root_sha256=rootsha)
    if not arrays:
        if ticket["split"]=="held":
            metadata["blind_forecast_sha256"]=blind_forecast(ticket,rootsha)
        result=dict(**metadata,status="skipped",reason=root["reason"],total_wall=time.perf_counter()-began,total_cpu=time.process_time()-cpu)
        save_json(pair_path,result);print(base,"skipped",root["reason"],flush=True);return result
    fp=io.BytesIO();np.savez_compressed(fp,**arrays)
    leafsha=publish(folder/(base+"_leaf.npz"),fp.getvalue())
    if ticket["split"]=="held":
        metadata["blind_forecast_sha256"]=blind_forecast(ticket,rootsha,leafsha)
    intervention={a["name"]:a["order"] for a in root["arms"]}
    intervention.update(menu_sha256=token(root["menu"]),cost=before_cost)
    archive=lambda name,x:save_json(folder/name,x)
    outcome=branches.continue_pair(ticket,intervention,ticket["arm_order"],restored,own,prefix,rng,archive)
    for a in outcome["arms"].values():
        a["terminal_value"]=mod._terminal(a["margin"])
    target=outcome["arms"]["first"]["terminal_value"]-outcome["arms"]["second"]["terminal_value"]
    result=dict(**metadata,status="complete",arms=outcome["arms"],target=target,leaf_sha256=leafsha,
                total_wall=time.perf_counter()-began,total_cpu=time.process_time()-cpu)
    save_json(pair_path,result);print(base,"complete",target,flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("split",choices=["train","held"]);parser.add_argument("--frozen-model",type=Path)
    opts=parser.parse_args()
    freight.check(sha(PLAN_BYTES)=="1d8b72c3eb39ac0d91119bacbdc3ec6912db66b6a14818da819e53fb5cd2d2c5","Changed action-pair panel")
    if opts.split=="held":
        freight.check(opts.frozen_model and opts.frozen_model.is_file(),"Missing training-only frozen model")
        save_json(HERE/"held_training_freeze.json",dict(protocol_sha256=sha(PLAN_BYTES),frozen_sha256=sha(opts.frozen_model.read_bytes())))
    freight.check(len(os.sched_getaffinity(0))==1,"Pin one CPU for paired continuation")
    os.environ["CARBON_DEBUG"]="1"
    all_results=[collect(t) for t in PLAN["tickets"] if t["split"]==opts.split]
    print(opts.split,[r["status"] for r in all_results],flush=True)
