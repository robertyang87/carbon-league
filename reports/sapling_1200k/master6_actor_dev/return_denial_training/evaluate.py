"""One-turn counterfactual outcome of the frozen public route decision, with unchanged teammate and enemy orders."""
from collections import Counter, deque
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from forecast import HERE, MOVES, NAMES, bind, extract, public_turn, require, restore, select
from ledger import one


def compact(obs):
    return {"t":obs["step"],"seat":obs["player"],"players":obs["players"]}


def selected_signature(g,t,c,o):
    identity=f"return-denial-1|{g['seed']}|{g['seat']}|{t}|{c['target']}|{o['own_uid']}|{o['order']}"
    return hashlib.sha256(identity.encode()).hexdigest()


def project(g,t,c,o,gain,sim,state,history,joint,cfg,base_result,branch,saved):
    seat=g["seat"]; target=c["target"]; uid=o["own_uid"]
    commands=[p.copy() for p in joint]
    if o["order"]=="NONE": commands[seat].pop(uid,None)
    else: commands[seat][uid]=o["order"]
    orig=joint[seat].get(uid) or "NONE"
    require(orig==o["old"] and orig!=o["order"] and commands[1-seat]==joint[1-seat],"alternative not a single changed own command")
    branch.set_state(saved)
    alt_result=extract.step(branch,state,cfg,commands)
    base=one(state,base_result[0],sim,base_result[1],seat,uid,target,joint[seat],joint[1-seat],cfg)
    alt=one(state,alt_result[0],branch,alt_result[1],seat,uid,target,commands[seat],commands[1-seat],cfg)
    net=alt["transfer_margin"]-base["transfer_margin"]
    dst=selected_signature(g,t,c,o)
    observed=joint[1-seat].get(target) or "NONE"
    pos=[dict(t=entry["step"],own=entry["players"][seat][2].get(uid),opponent=entry["players"][1-seat][2].get(target))
         for entry in history if t-entry["step"] in (3,5,10)]
    public_info={k:c[k] for k in ("target","origin","cargo","return_distance","action_cells","shortest","p")}
    return dict(signature=dst,t=t,seat=seat,seed=g["seed"],target=public_info,choice=o,history=pos,estimated_extra=gain,
                issued=gain>0,observed_enemy_command=observed,observed_enemy_cell=c["action_cells"][MOVES.index(observed)],
                enemy_command_is_label_only=True,v1=base,alternative=alt,delta_transfer=net,
                delta_cash_margin=(alt["cash_change"][seat]-alt["cash_change"][1-seat])-(base["cash_change"][seat]-base["cash_change"][1-seat]),
                delta_own_harvest=alt["real_new_harvest"][seat]-base["real_new_harvest"][seat],
                delta_own_deposit=alt["cash_entries"][seat].get("own_delivery",0)-base["cash_entries"][seat].get("own_delivery",0),
                delta_open_shortest=len(o["open_shortest_new"])-len(o["open_shortest_old"]))


def run_game(g,model,split):
    raw,cfg,sim=restore(g); branch=extract.FastCarbon(cfg)
    seat=g["seat"];history=deque(maxlen=10);counter=Counter();rows=[];forecasts=[]
    if raw.get("errors") or raw.get("examiner_md5")!="57f9a594cce4f2ec9e3573d8d9d26ca1" or not any(x[1]=="searched" for x in raw["search_calls"]):
        raise ValueError("unregistered examiner or unreliable native source")
    for t,joint in enumerate(raw["actions"]):
        obs=sim.obs(seat)
        carriers=public_turn(obs,list(history),joint[seat],cfg)
        choices=select(carriers,model,cfg)
        counter["source_turns"]+=1
        counter["immediate_returns"]+=len(carriers)
        counter["possible_actions_different"]+=sum(len(c["options"]) for c in carriers)
        counter["no_eligible_enemy_turns"]+=not carriers
        counter["return_but_no_interceptor_turns"]+=bool(carriers) and choices["forecast"] is None
        counter["actionable_turns"]+=choices["forecast"] is not None
        for c in carriers:
            observed=joint[1-seat].get(c["target"]) or "NONE"
            i=MOVES.index(observed)
            forecasts.append(dict(t=t,uid=c["target"],predicted=c["p"],physical_label=observed,original_cargo=c["cargo"],
                                  home_distance=c["return_distance"],shortest=c["shortest"],possible_pairs=len(c["options"])))
        state=sim.full_observation()
        saved=sim.get_state() if choices["forecast"] is not None else None
        factual=extract.step(sim,state,cfg,joint)
        if saved is not None:
            arms={}
            cached={}
            for name, (c,o,gain) in choices.items():
                key=c["target"],o["own_uid"],o["order"]
                if key not in cached:
                    cached[key]=project(g,t,c,o,gain,sim,state,history,joint,cfg,factual,branch,saved)
                arms[name]={**cached[key],"estimated_extra":gain,"issued":gain>0}
            rows.append(dict(t=t,arms=arms))
        history.append(obs)
    require([round(x,2) for x in sim.cash]==g["scores"],"original score reconciliation")
    source=extract.ROOT/g["raw_path"]
    doc=dict(map=g["seed"],seat=seat,source_tape=g["raw_path"],source_sha256=g["raw_sha256"],board_sha256=g["map_sha256"],
             factual_scores=g["scores"],search_counts=dict(Counter(r[1] for r in raw["search_calls"])),counts=dict(counter),
             rows=rows,forecasts=forecasts,source_terminal_checked=True,weights_sha256=extract.sha(HERE/"weights.json"))
    dest=HERE/f"{split}_{g['seed']}_{seat}.json.gz"
    extract.write(dest,doc)
    print(split,g["seed"],seat,dict(counter),flush=True)


if __name__=="__main__":
    split=sys.argv[1]
    spec=bind(split)
    model=extract.load(HERE/"weights.json")
    require(model["feature_code_sha256"]==extract.sha(HERE/"policy.py") and set(map(int,model["fit_map_counts"]))==set(bind("train")["maps"]),"changed model feature source or training maps")
    for game in spec["tapes"]:
        run_game(game,model,split)
