"""Independent deployment trace on held-out maps; alternatives change just this legal physical order."""
import gzip,json,importlib.util,os,sys
from pathlib import Path
from collections import Counter
from train import HERE,ROOT,TR,sha,games
from route import N,MOVE
from extract import step,FastCarbon

def write(path,obj):
    b=json.dumps(obj,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
    data=gzip.compress(b,mtime=0) if path.suffix==".gz" else b
    tmp=path.with_suffix(".tmp");tmp.write_bytes(data);os.replace(tmp,path)

def focal(before,after,sim,ev,seat,row,order,cfg,joint):
    uid=row["uid"]; origin,cargo,_=before["players"][seat][2][uid]
    pos=N[origin][MOVE.index(order)]
    initial=cargo*(1-cfg["moveCost"])
    credits=sum(load for wid,_,load,_,winner in sim.last_collisions if winner==seat and any(
        wid in group[2] and N[group[2][wid][0]][MOVE.index(orders.get(wid) or "NONE")]==pos
        for group,orders in zip(before["players"],joint)))
    final=after["players"][seat][2].get(uid)
    if final:
        if final[0]!=pos or initial+credits<=0:raise ValueError("worker identification")
        fraction=final[1]/(initial+credits)
        if not -1e-7<=fraction<=1+1e-7:raise ValueError(("root cargo not conserved",initial,credits,final))
        retained=initial*fraction
        tax=initial-retained
    else:
        retained=0.;tax=0.
    if any(e[0] in ("own_delivery","opponent_worker_delivery_to_this_base") and e[3]==uid for e in ev):
        raise ValueError("candidate changed a deposit")
    return dict(retained_root=retained,paid_tax_from_root=tax,alive=final is not None,acquired_from_collisions=credits,
                income_both_before_rounding=[sum(e[2] for e in ev if e[1]==p and e[0] not in
                ("plant","seize","hire_collector","hire_planter","rounding")) for p in (0,1)],
                team_tree_absorption=[sum(e[2] for e in ev if e[1]==p and e[0]=="tree_absorption") for p in (0,1)],
                new_harvest_by_focal=0.,delivered_root_this_step=0.)

def load_candidate():
    spec=importlib.util.spec_from_file_location("rv_frozen_agent",HERE/"candidate.py")
    agent=importlib.util.module_from_spec(spec); spec.loader.exec_module(agent)
    return agent

def run():
    release=json.loads((HERE/"release.json").read_bytes())
    if any(sha(HERE/k)!=h for k,h in release["sha256"].items()): raise ValueError("frozen sources disagree")
    agent=load_candidate()
    totals={};branches=[];allrows=[]
    reg=json.loads((HERE/"validation_registration.json").read_bytes())
    if reg["candidate_sha256"]!=sha(HERE/"candidate.py") or reg["initial_protocol_sha256"]!=sha(HERE/"protocol.json"):
        raise ValueError("validation registration drift")
    def registered_games():
        for part in reg["validation"]:
            if sha(ROOT/part["receipt"])!=part["sha256"]:raise ValueError("validation receipt drift")
            official=json.loads((ROOT/part["receipt"]).read_bytes())
            for info in part["games"]:
                path=ROOT/info["raw_path"]
                if sha(path)!=info["raw_sha256"]:raise ValueError("raw tape drift")
                source=next(g for g in official["games"] if g["seed"]==info["seed"] and g["seat"]==info["seat"])
                yield dict(**info,official_cash=source["scores"]),json.loads(gzip.decompress(path.read_bytes()))
    for identity,raw in registered_games():
        seed,s=identity["seed"],identity["seat"]
        out=HERE/f"validation_{seed}_{s}.json.gz"
        expected=dict(source=identity["raw_sha256"],candidate=release["sha256"]["candidate.py"],evaluator=sha(Path(__file__)))
        if out.exists():
            item=json.loads(gzip.decompress(out.read_bytes()))
            if item["identity"]!=expected:raise ValueError("stale per-game receipt")
        else:
            sim=FastCarbon(raw["game_configuration"]).reset(seed=seed);cfg=sim.config
            alt=FastCarbon(cfg); evidence=[];count=Counter()
            for t,j in enumerate(raw["actions"]):
                public=sim.obs(s)
                issued=agent.agent(public,cfg);trace=agent._RV_TRACE
                if trace["base"]!=j[s]:raise ValueError(("parent action drift",seed,s,t))
                count["agent_calls"]+=1;count["geometric_options"]+=trace["ranked"]
                row=trace["selected"]
                if not row:
                    if issued!=j[s]:raise ValueError("unreported edit")
                    sim.step(*j);continue
                obs=sim.full_observation();state=sim.get_state()
                alternative=list(j);alternative[s]=issued
                if issued!={**j[s],row["uid"]:row["alternative"]}:raise ValueError("not single legal command")
                before,ev=step(sim,obs,cfg,j)
                baseline=focal(obs,before,sim,ev,s,row,row["old"],cfg,j)
                alt.set_state(state);after,ev=step(alt,obs,cfg,alternative)
                changed=focal(obs,after,alt,ev,s,row,row["alternative"],cfg,alternative)
                count["edits"]+=1
                count["retained_root_gain"]+=changed["retained_root"]-baseline["retained_root"]
                count["focal_new_deaths"]+=baseline["alive"] and not changed["alive"]
                count["focal_deaths_prevented"]+=not baseline["alive"] and changed["alive"]
                evidence.append(dict(t=t,seat=s,seed=seed,**row,parent=baseline,candidate=changed))
            if sim.step_count!=identity["steps"] or list(sim.cash)!=identity["official_cash"]:raise ValueError("replay drift")
            item=dict(identity=expected,count=dict(count),rows=evidence,final_cash=list(sim.cash))
            write(out,item)
        totals.setdefault(seed,Counter()).update(item["count"]); allrows.extend(item["rows"])
    allowed=all(c["retained_root_gain"]>1e-6 and c["focal_new_deaths"]==0 for c in totals.values())
    for seed in sorted(totals):
        rows=sorted([r for r in allrows if r["seed"]==seed],key=lambda r:(r["t"],r["seat"]))
        if rows:branches.append(rows[0])
    write(HERE/"validation.json",dict(candidate_sha256=release["sha256"]["candidate.py"],maps=totals,
        planned_native_gate_pass=allowed,first_enrolled=branches,selection="first chronological actual trigger per map (seat0 if tied)",
        scope="Archived trajectory, paired one simultaneous action; original source replay terminal. Full candidate policy has not played any terminal matches here."))
if __name__=="__main__":run()
