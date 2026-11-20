"""Freeze successor estimates using registered simultaneous root orders, independently cloned histories."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
from learn import HERE,INTAKE,History,check,common,digest,predict,save,selection
from zerosum_env.envs.carbon import idgen


def measure(ticket,model):
    seat=ticket["seat"]
    root=ticket["root_step"]
    history=History(seat)
    source,meta=selection.source_for(ticket["seed"])
    check(meta["source_sha256"]==ticket["source_sha256"] and meta["terrain"]==ticket["terrain"],"source prefix drift")
    cfg=source["configuration"]
    env=common.collect.begin(ticket["seed"],cfg)
    check(hashlib.sha256(bytes(env.states[0].observation.full_carbon)).hexdigest()==ticket["terrain"],"different initial terrain")
    for step,r in enumerate(source["rows"][:root+1]):
        obs=r["public"]
        check(step==obs["step"] and seat==obs["player"],"prefix ordering")
        view=common.collect.public(env,seat)
        view["remainingOverageTime"]=obs["remainingOverageTime"]
        check(common.digest(view)==common.digest(obs),"public state reconstruction")
        history.observe(obs)
        if step<root:
            env.step(r["orders"],[])
    check(common.digest(obs)==ticket["root_public_sha256"] and not env.done,"unsigned root")
    realized_opponent=r["orders"][1-seat]
    ids_before=copy.deepcopy(idgen._id_dict)
    result={}
    for name in ("parent","alternative"):
        successor=copy.copy(env)
        successor.states=copy.deepcopy(env.states)
        successor.steps=env.steps[:-1]+[successor.states]
        successor.logs=[]
        successor.error=env.error.copy()
        ids=idgen._id_dict
        ids.clear();ids.update(ids_before)
        joint=r["orders"].copy()
        joint[seat]=ticket["orders"][name]
        successor.step(joint,[])
        branch=copy.deepcopy(history)
        next_obs=common.collect.public(successor,seat)
        x,cash,h=branch.observe(next_obs)
        value=float(predict(model,x[None],np.array([h]),np.array([cash]))[0])
        if successor.done:
            score=[s.reward for s in successor.states]
            value=float(score[seat]-score[1-seat])
        result[name]=dict(team_order=joint[seat],after_step=int(next_obs["step"]),
                          after_cash=[p[0] for p in next_obs["players"]],cash_margin=cash,
                          predicted_official_margin=value,native_public_sha256=common.digest(next_obs),
                          feature_sha256=hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(),
                          exact_terminal=bool(successor.done))
    idgen._id_dict.clear();idgen._id_dict.update(ids_before)
    delta=result["alternative"]["predicted_official_margin"]-result["parent"]["predicted_official_margin"]
    drift=result["alternative"]["cash_margin"]-result["parent"]["cash_margin"]
    sign=lambda v: "tie" if abs(v)<=1e-7 else ("alternative" if v>0 else "parent")
    return dict(seed=ticket["seed"],terrain=ticket["terrain"],seat=seat,step=root,root_public_sha256=ticket["root_public_sha256"],
                conditional_on_actual_simultaneous_opponent=realized_opponent,arms=result,delta=delta,cash_delta=drift,
                pick=sign(delta),cash_pick=sign(drift))


def main():
    fit=json.loads((HERE/"fit.json").read_bytes())
    check(fit["weights"]==digest(HERE/"weights.npz") and fit["protocol"]==digest(HERE/"protocol.json"),"weight or protocol change")
    roots=selection.read_json(INTAKE/"roots.json")
    archived=selection.read_json(INTAKE/"prediction/registered.json")
    check(archived["roots"]==digest(INTAKE/"roots.json"),"different archived critic roots")
    with np.load(HERE/"weights.npz",allow_pickle=False) as w:
        model={k:w[k] for k in w}
    rows=[measure(r,model) for r in roots["tickets"]]
    for r in rows:
        comp,=[a for a in archived["rows"] if a["seed"]==r["seed"] and a["seat"]==r["seat"]]
        r["existing_action_critic"]={"has_both_actions":comp["pair_score_difference"] is not None,
             "pick":comp["higher_arm"],"own_score_units_delta":comp["pair_score_difference"]}
    check(len(rows)==4 and {r["terrain"] for r in rows}.isdisjoint(c["terrain"] for c in fit["source_games"]),
          "map split leakage")
    save(HERE/"frozen.json",dict(fit=digest(HERE/"fit.json"),weights=fit["weights"],protocol=fit["protocol"],
         trainer=fit["trainer"],scoring_code=digest(HERE/"forecast.py"),root_source=digest(INTAKE/"roots.json"),
         archived_critic=digest(INTAKE/"prediction/registered.json"),rows=rows))
    print(datetime.now(timezone.utc).isoformat(),digest(HERE/"frozen.json"),flush=True)


if __name__=="__main__":
    main()
