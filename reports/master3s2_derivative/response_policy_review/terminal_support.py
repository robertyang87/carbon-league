"""Support vs. ordering on archived last decisions facing the original opponent. No continuation games."""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
D=HERE.parent
REPO=D.parents[1]
sys.path.insert(0,str(D/"opponent_policy"))
from original_policy import module
spec=importlib.util.spec_from_file_location("support_referee",D/"freight_response/run.py")
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(p):
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix==".gz" else b)


def key(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)


def check(ok,why):
    if not ok:raise ValueError(why)


def public(model,obs,cfg,tracker,previous,ticket,cost):
    seed=int.from_bytes(hashlib.sha256(("last-turn-support/v1/"+ticket["sha256"]).encode()).digest()[:4],"big")
    model._ST=dict(tr=tracker,prev=previous,rng=np.random.RandomState(seed))
    model._S_COST[:]=cost
    capture={}
    def receive(code,offset,result):
        capture.update(sys._getframe(1).f_locals)
    m=sys.monitoring;slot=4;m.use_tool_id(slot,"observe native root menu")
    try:
        m.register_callback(slot,m.events.PY_RETURN,receive);m.set_local_events(slot,model._search.__code__,m.events.PY_RETURN)
        start=time.perf_counter();issued=model.agent(obs,cfg);elapsed=time.perf_counter()-start
    finally:
        m.set_local_events(slot,model._search.__code__,0);m.register_callback(slot,m.events.PY_RETURN,None);m.free_tool_id(slot)
    menu=capture.get("cands",[issued]);check(issued in menu,"Native public order not in its menu")
    return issued,menu,dict(K=len(menu),wall_s=elapsed,depth=capture.get("depth",0),incoming_cost=cost,policy_rng=seed)


def run(card,model,cost):
    raw=(REPO/card["path"]).read_bytes();check(sha(raw)==card["sha256"],"Archive")
    tape=json.loads(gzip.decompress(raw));s=card["candidate_seat"];foe=1-s;cfg=tape["game_configuration"]
    actions=tape["actions"];check(len(actions)==299 and card["expert_seat"]==foe and card["transitions"]==299
                                  and sha(key(actions).encode())==card["actions_sha256"],"Signed sequence")
    random.seed(card["seed"]);np.random.seed(card["seed"])
    env=ref.make("carbon",configuration={**cfg,"randomSeed":card["seed"]})
    sim=ref.econ.FastCarbon(cfg).reset(seed=card["seed"])
    check(sha(bytes(env.states[0].observation.full_carbon))==card["terrain"]==sha(bytes(map(int,sim.carbon))),"Terrain")
    tracker=model.BeliefTracker(model._Bel,s);previous=None;cash=[]
    for step in range(299):
        board=env.states[0].observation
        check(not env.done and board.step==step,"Unexpected terminal")
        public_carbon=env.states[s].observation.carbon
        cash.append([p[0] for p in board.players])
        if step==298:break
        tracker.step(board.players,step,public_carbon)
        tracker.set_actions(board.players,actions[step][s]);previous=board.players
        env.step(actions[step],[]);sim.step(*actions[step])
        check(list(sim.cash)==[p[0] for p in env.states[0].observation.players],"Physical prefix")
    obs=dict(step=298,player=s,carbon=public_carbon,players=board.players,trees=board.trees,remainingOverageTime=env.states[s].observation.get("remainingOverageTime",60))
    issued,menu,stats=public(model,obs,{k:v for k,v in cfg.items() if k!="randomSeed"},tracker,previous,card,cost)
    other=actions[298][foe];archive_order=actions[298][s]
    # The original archive never stored the alternate own moves. Evaluate them in the signed official one-step referee.
    # Only the current state, history length, id counters and current RNG enter this last transition.
    env.steps[:-1]=[None]*298;env.logs=[]
    frozen=ref.official_state.take(env,[0,0]);bank0=[x[0] for x in board.players];q={}
    def score(order):
        signature=key(order)
        if signature in q:return q[signature]
        ref.official_state.restore(env,frozen);aux=copy.deepcopy(sim)
        pair=[{},{}];pair[s]=order;pair[foe]=other
        env.step(pair,[]);aux.step(*pair)
        after=env.states[0].observation;rewards=[p.reward for p in env.states]
        check(env.done and [x.status for x in env.states]==["DONE","DONE"] and [x[0] for x in after.players]==list(aux.cash) and
              after.players==aux.obs(0)["players"] and after.trees==aux.obs(0)["trees"] and after.full_carbon==aux.carbon,"One-step referee comparison")
        extinct=[not p[2] and not p[3] and (not p[1] or p[0]<min(cfg["recCollectorCost"],cfg["recPlanterCost"])) for p in after.players]
        independent=[299-cfg["episodeSteps"]-1 if extinct[k] else (bank0[k] if any(extinct) else aux.cash[k]) for k in range(2)]
        check(independent==rewards,"Final reward convention")
        q[signature]=round(rewards[s]-rewards[foe],2);return q[signature]
    score(archive_order);check([p.reward for p in env.states]==tape["official_score_by_seat"] and sha(key(cash+[[p[0] for p in env.states[0].observation.players]]).encode())==card["official_cash_path_sha256"],"Archive score/cash-path match")
    one=[]
    for k in list(board.players[s][2])+list(board.players[s][1]):
        possible=("NONE","UP","DOWN","LEFT","RIGHT") if k in board.players[s][2] else ("NONE","RECCOLLECTOR","RECPLANTER")
        for move in possible:
            if move==issued.get(k,"NONE"):continue
            a=issued.copy()
            if move=="NONE":a.pop(k,None)
            else:a[k]=move
            one.append(a)
    native_best=max(menu,key=score);extra_best=max([native_best,*one],key=score);base=score(issued);in_best=score(native_best);expanded=score(extra_best)
    native_set={key(a) for a in menu}
    return dict(terrain=card["terrain"],seed=card["seed"],side=s,split=card["split"],archive=card["sha256"],card=card["path"],
                rank_case=stats,opponent_order=other,original_wire_order=archive_order,original_wire_margin=score(archive_order),
                issued=issued,issued_margin=base,native_oracle_order=native_best,native_oracle_margin=in_best,expanded_oracle_order=extra_best,expanded_oracle_margin=expanded,
                one_edits=len(one),outside_one_edits=sum(key(a) not in native_set for a in one),
                outside_one_edits_strictly_better_than_best_native=sum(key(a) not in native_set and score(a)>in_best for a in one),
                distinct_official_branches=len(q),ranking_regret=round(in_best-base,2),support_regret=round(expanded-in_best,2))


def main():
    import torch
    os.environ["CARBON_DEBUG"]="1";torch.set_num_interop_threads(1)
    plan=D/"opponent_policy/protocol.json";cards=read(plan)["games"];raw=(D/"cold_start/candidate.py").read_bytes()
    check(sha(raw)=="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed" and os.sched_getaffinity(0)=={3},"Source or CPU")
    model=module(raw);torch.set_num_threads(1);cost=list(model._S_COST)
    out=[];cache=Path("/tmp/dspro-rr-support-"+sha(Path(__file__).read_bytes())[:10]);cache.mkdir(exist_ok=True)
    for i,card in enumerate(cards):
        p=cache/(card["sha256"]+".json")
        if p.exists():row=read(p)
        else:
            row=run(card,model,cost);p.write_text(json.dumps(row,sort_keys=True))
        out.append(row)
        if (i+1)%8==0:print("archived roots",i+1,flush=True)
    dst=HERE/"terminal_support.json";doc=dict(source_sha256=sha(raw),protocol_sha256=sha(plan.read_bytes()),method_sha256=sha(Path(__file__).read_bytes()),
          exact_label="Intervene on candidate seat final command; opposing seat is historical natural full original Master3s2 simultaneously issued command. One official remaining transition, no opponent continuation.",
          acquisition="Public V2 action from archived student-side observation histories, newly frozen private draw per signed case and same measured cold-cache cost estimates. The registered source archive was originally generated by older student policies versus native original; these histories are off-policy for V2.",rows=out)
    b=(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    check(not dst.exists() or dst.read_bytes()==b,"Prior trial differs");dst.write_bytes(b)


if __name__=="__main__":main()
