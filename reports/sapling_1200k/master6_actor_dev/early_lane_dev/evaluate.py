"""Original-route predictions and independent true one-step overrides, kept distinct from induced route changes."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

import chooser
from train import HERE, PROTOCOL, batch, check, distance, extract, forecast, load_tape, sources_at, target


def reference(cells,origin,bases):
    current=min(distance(origin,b) for b in bases)
    points={c:math.exp(.70*(current-min(distance(c,b) for b in bases))-.65*distance(origin,c)) for c in cells if c!=-1}
    z=sum(points.values())
    return {**{c:.80*v/z for c,v in points.items()},-1:.20}


def objective(p,y):
    return {"brier":sum(v*v for v in p.values())-2*p[y]+1,"correct":max(p,key=p.get)==y,
            "logloss":-math.log(max(1e-15,p[y])),"other":y==-1,"p_other":p[-1]}


def reachable(arm,which,y):
    if which=="alt":
        endpoints,available=arm["cells"],arm["reachable_alt"]
    else:
        endpoints,available=arm["old_best_cells"],arm["old_best_reachable"]
    if len(endpoints)!=2:return 0.
    return .5*sum(int(cell==hit and cell in available) for cell,hit in zip(endpoints,y))


def game(g,net):
    raw,obs,cash,states,original_deposits=load_tape(g,save_states=True)
    cfg=raw["game_configuration"];seat=g["seat"];sim=extract.FastCarbon(cfg)
    counts=Counter()
    ss={str(h):{"model":Counter(),"baseline":Counter()} for h in (3,5)}
    forecast_records=[];decisions=[]
    for t,joint in enumerate(raw["actions"]):
        current=obs[t];history=obs[max(0,t-10):t]
        actual=joint[seat]; forecasts={}; endpoints={}
        if t+5 < len(obs):
            for uid in sources_at(current):
                en=current["players"][1-seat]; origin=en[2][uid][0]
                forecasts[uid]={}
                endpoints[uid]=[]
                for h in (3,5):
                    ph=forecast.predict(net,current,history,uid,h,cfg)
                    check(-1 in ph and abs(sum(ph.values())-1.)<1e-8,"incomplete probability mass")
                    forecasts[uid][h]=ph
                    baseline=reference(ph,origin,en[1].values())
                    y=target(obs,cash,t,uid,h,seat)
                    check(y in ph,"missing true class")
                    endpoints[uid].append(y)
                    for k,probs in (("model",ph),("baseline",baseline)):
                        sc=objective(probs,y)
                        ss[str(h)][k].update(sc);ss[str(h)][k]["cases"]+=1
                    forecast_records.append([t,uid,h,y,max(ph,key=ph.get),ph[y],ph[-1],objective(ph,y)["brier"],objective(baseline,y)["brier"]])
        # The only inputs available to the chooser are forecasts, our own planned order and this view.
        decision=chooser.choose(current,actual,cfg,forecasts)
        a=decision["candidate"]; out=decision["commands"]
        delta=[u for u in (set(out)|set(actual)) if (out.get(u) or "NONE")!=(actual.get(u) or "NONE")]
        counts["source_turns"]+=1;counts["eligible_heavy_workers"]+=len(forecasts)
        counts["turn_with_forecast"]+=bool(forecasts)
        if a is None:
            check(not delta and not decision["triggered"],"unregistered change on an empty turn")
            counts["no_candidate"]+=1
            continue
        uid=a["own_uid"];enemy=a["target_uid"]; y=endpoints[enemy]
        fire=decision["triggered"]
        check((delta==[uid]) if fire else not delta,"package must affect one actual order only")
        check(a["old"]==(actual.get(uid) or "NONE") and a["old"]!=a["order"],"registered root differs from actor")
        v1_hit=reachable(a,"v1",y);alt_hit=reachable(a,"alt",y)
        counts["candidate"]+=1;counts["issued"]+=fire
        # A non-issued candidate is retained for sparse response follow-up and diagnostics, never credited.
        sim.set_state(states[t])
        state=sim.full_observation()
        override=[orders.copy() for orders in joint]
        if a["order"]=="NONE":override[seat].pop(uid,None)
        else:override[seat][uid]=a["order"]
        next_state,events=extract.step(sim,state,cfg,override)
        enemy_after=next_state["players"][1-seat][2].get(enemy)
        focal_after=next_state["players"][seat][2].get(uid)
        v1_after=obs[t+1]["players"][seat][2].get(uid)
        altpaid=sum(e[2] for e in events if e[0]=="own_delivery" and e[1]==seat and e[3]==uid)
        oldpaid=original_deposits[t].get(uid,0.)
        harm=v1_after is not None and focal_after is None
        saved=v1_after is None and focal_after is not None
        check(focal_after is None or focal_after[0]==a["cell"],"applied move did not reach registered first cell")
        gain=alt_hit-v1_hit
        if fire:
            counts["trace_gain_sum"]+=gain
            counts["trace_gain_positive"]+=gain>0
            counts["trace_gain_negative"]+=gain<0
            counts["trace_hit_alternative"]+=alt_hit
            counts["trace_hit_v1"]+=v1_hit
            counts["harm"]+=harm;counts["save"]+=saved;counts["lost_factual_delivery"]+=altpaid<oldpaid-1e-7
            counts["normal_deposit_difference"]+=altpaid-oldpaid
            counts["entered_target_cell_at_3_or_5"]+=any(y[i]==a["cells"][i] for i in range(2))
        sig=hashlib.sha256(f"early-lane-v1|{g['seed']}|{seat}|{t}|{uid}|{enemy}|{a['order']}".encode()).hexdigest()
        decisions.append(dict(t=t,seed=g["seed"],seat=seat,signature=sig,candidate=a,issued=fire,original_trace_endpoints=y,
                              v1_trace_coverage=v1_hit,alt_trace_coverage=alt_hit,trace_gain=gain,
                              registered_single_step_delta=dict(harmed=harm,saved=saved,normal_delivery_v1=oldpaid,normal_delivery_alternative=altpaid,
                                                               target_survived=enemy_after is not None,mover_survived=focal_after is not None),
                              history=[{"t":k,"own":obs[k]["players"][seat][2].get(uid),"enemy":obs[k]["players"][1-seat][2].get(enemy)}
                                       for k in (t-10,t-5,t-3,t) if k>=0]))
    return dict(seed=g["seed"],seat=seat,steps=g["steps"],source=g["raw_path"],source_sha256=g["raw_sha256"],score=g["scores"],
                counts=dict(counts),scores=ss,forecast_schema=["t","uid","h","actual","top_class","p_actual","p_other","brier_fit","brier_base"],
                forecasts=forecast_records,decisions=decisions,weights_sha256=extract.sha(HERE/"weights.json"),
                chooser_sha256=extract.sha(HERE/"chooser.py"),registered_source_and_counterfactuals_independently_checked=True)


def main(batch_no):
    specs=batch(batch_no)
    net=extract.load(HERE/"weights.json")
    check(net["protocol_sha256"]==PROTOCOL and net["forecast_sha256"]==extract.sha(HERE/"forecast.py"),"wrong predictor")
    for g in specs:
        out=game(g,net);path=HERE/f"validation_{g['seed']}_{g['seat']}.json.gz"
        extract.write(path,out);print(path.name,out["counts"],flush=True)


if __name__=="__main__":
    main(int(sys.argv[1]))
