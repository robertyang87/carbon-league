"""Audit the untouched original critic from opposite seats at registered physical positions."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from collections import defaultdict
from statistics import mean

for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="1"

import numpy as np
from register import HERE, build

sys.path.insert(0,str(HERE.parents[1]/"opponent_policy"))
import original_policy as original
from original_policy import need, dump, free_entry

PROTOCOL="9cb5380ec8c4c59043d55778a7666095ec9f17ae5173e431736c5764ddf9f5f6"
SIDES=("self","opposite","pair_mean")


def forward(m,net,players,previous,carbon,step,seat):
    seats=(seat,1-seat)
    x=np.stack([m._quant(m.encode(players,step,q,carbon,[1.]*m.N,previous)) for q in seats]).astype(np.float32)
    scal=np.stack([m._scalars(players,step,q) for q in seats])
    mask=np.array([-1.,-1.,1.,-1.,-1.],np.float32)
    need(x.shape==(2,54,15,15) and scal.shape==(2,5) and np.array_equal(scal[0]*mask,scal[1]),"Wrong seat/scalars")
    _,_,values=net(x,m._SID,scal)
    v,other=map(float,np.asarray(values).reshape(2))
    need(np.isfinite([v,other]).all(),"Nonfinite leaf")
    return dict(self=v,opposite=-other,pair_mean=(v-other)/2.,raw_other=other,signed_inconsistency=v+other,
                input_sha256=[original.sha(a.tobytes()) for a in x],scalars=scal.tolist())


def game(card,m,net):
    tape=original.tape(card)
    seat=card["expert_seat"]
    random.seed(card["seed"])
    np.random.seed(card["seed"])
    env=free_entry.official_rules(free_entry.ROOT)("carbon",configuration={**tape["game_configuration"],"randomSeed":card["seed"]})
    terrain=original.sha(bytes(env.states[0].observation.full_carbon))
    need(terrain==card["terrain"],"Opening mismatch")
    tracker=m.BeliefTracker(m._Bel,seat)
    previous=None
    prediction=None
    cash=[]
    for t,commands in enumerate(tape["actions"]):
        view=env.states[0].observation
        players=view.players
        need(not env.done and t==view.step,"Archive crosses terminal or skips a turn")
        cash.append([p[0] for p in players])
        if t<=card["step"]:
            private=env.states[seat].observation
            estimate,_=tracker.step(players,t,private.carbon)
            if t==card["step"]:
                prediction={"estimated":forward(m,net,players,previous,estimate,t,seat),
                            "simulator_full_control":forward(m,net,players,previous,view.full_carbon,t,seat)}
                prediction["shared_estimate_sha256"]=original.sha(np.asarray(estimate,"<f8").tobytes())
                prediction["hidden_carbon_mae"]=float(np.mean(abs(np.asarray(estimate)-np.asarray(view.full_carbon))))
            tracker.set_actions(players,commands[seat])
        previous=players
        env.step(commands,[])
    cash.append([p[0] for p in env.states[0].observation.players])
    scores=[r.reward for r in env.states]
    need(env.done and scores==tape["official_score_by_seat"] and
         free_entry.digest(cash)==card["official_cash_path_sha256"],"Official termination or cash path differs")
    margin=scores[seat]-scores[1-seat]
    target=m._terminal(margin)
    need(target==-m._terminal(-margin),"Zero-sum target violated")
    return dict(seed=card["seed"],terrain=terrain,scope=card["split"],expert_seat=seat,step=card["step"],
                source_sha256=card["sha256"],official_scores_by_seat=scores,margin=margin,target=target,
                transitions=len(tape["actions"]),available=prediction is not None,prediction=prediction)


def describe(rows):
    out={}
    for kind in ("estimated","simulator_full_control"):
        maps=defaultdict(list)
        for r in rows:
            if r["available"]:
                maps[r["terrain"]].append(r)
        def by_map(f):
            return {terrain:mean(f(r,r["prediction"][kind]) for r in rr) for terrain,rr in maps.items()}
        def metric(f):
            return mean(by_map(f).values())
        metrics={}
        for name in SIDES:
            errors=by_map(lambda r,d:(d[name]-r["target"])**2)
            metrics[name]=dict(rmse=mean(errors.values())**.5,mean_squared_error=mean(errors.values()),
                               mae=metric(lambda r,d:abs(d[name]-r["target"])),
                               bias=metric(lambda r,d:d[name]-r["target"]),
                               sign_accuracy=metric(lambda r,d:float(np.sign(d[name])==np.sign(r["target"]))),
                               by_map_mse=errors)
        gains={k:v-metrics["pair_mean"]["by_map_mse"][k] for k,v in metrics["self"]["by_map_mse"].items()}
        out[kind]=dict(registered=len(rows),available=sum(len(rr) for rr in maps.values()),maps=len(maps),
                       signed_mean_seat_sum=metric(lambda r,d:d["signed_inconsistency"]),
                       mean_abs_seat_sum=metric(lambda r,d:abs(d["signed_inconsistency"])),
                       more_accurate_than_self_maps=sum(v>0 for v in gains.values()),self_minus_pair_mse=mean(gains.values()),
                       by_map_mse_gain=gains,estimators=metrics)
    return out


if __name__=="__main__":
    raw=(HERE/"protocol.json").read_bytes();plan=json.loads(raw)
    need(original.sha(raw)==PROTOCOL and plan==build(),"Wrong frozen roots")
    model=original.module(original.original())
    model._torch.set_num_threads(1)
    net=model._net()
    rows=[game(c,model,net) for c in plan["samples"]]
    dump(HERE/"positions.json.gz",rows)
    dump(HERE/"results.json",dict(protocol_sha256=PROTOCOL,executor_sha256=original.sha(Path(__file__).read_bytes()),
                                 source_sha256=plan["examiner"],rows_sha256=original.sha((HERE/"positions.json.gz").read_bytes()),
                                 seat_oracle_separate=True,by_split={k:describe([r for r in rows if r["scope"]==k]) for k in ("train","val")}))
