"""Exact one-transition alternatives for ds-pro-v2, last 30 public decisions against Fable V7."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
D=HERE.parents[2]
spec=importlib.util.spec_from_file_location("archived_official",D/"freight_response/run.py")
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
ACTIONS=("NONE","UP","RIGHT","DOWN","LEFT")
BASES=("NONE","RECCOLLECTOR","RECPLANTER")
SOURCE=HERE.parents[1]/"replays/bdbe59b04f71e43e53696874c8fd5958-1.json.gz"
ME=1


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(data):
    return sha(json.dumps(data,sort_keys=True,separators=(",",":")).encode())


def check(ok,msg):
    if not ok:raise ValueError(msg)


def physical(state,archive=False):
    obs=[s["observation"] for s in state] if archive else [s.observation for s in state]
    v=obs[0]
    return [v["step"],v["players"],v["trees"],v["full_carbon"],[s["carbon"] for s in obs]]


def cmd(base,id,choice):
    a=base.copy()
    if choice=="NONE":a.pop(id,None)
    else:a[id]=choice
    return a


def investigate():
    raw=SOURCE.read_bytes();game=json.loads(gzip.decompress(raw));tape=game["steps"];cfg=game["configuration"]
    random.seed(cfg["randomSeed"]);ref.np.random.seed(cfg["randomSeed"])
    env=ref.make("carbon",configuration=cfg);opening=tape[0][0]["observation"]["full_carbon"]
    env.states[0].observation.full_carbon=copy.deepcopy(opening)
    sim=ref.econ.FastCarbon(cfg).reset(full_carbon=opening);sim.cash=ref.econ.CashEvents(sim.cash)
    def compare(engine,states):
        o=engine.obs(0)
        fast=[engine.step_count,o["players"],o["trees"],list(engine.carbon),[engine.obs(k)["carbon"] for k in (0,1)]]
        check(fast==physical(states),"Fast and official disagree")
    findings=[]; intercept={}; final_two={}
    for t in range(299):
        check(physical(env.states)==physical(tape[t],True),f"Historical state {t}")
        compare(sim,env.states)
        actions=[r["action"] for r in tape[t+1]]
        if t>=269:
            env.steps[:-1]=[None]*t;env.logs=[]
            frozen=ref.official_state.take(env,[0,0])
            root=env.states[0].observation;players=copy.deepcopy(root.players)
            def settle(a):
                ref.official_state.restore(env,frozen);s=copy.deepcopy(sim);s.cash.events.clear()
                s.step(*a);env.step(a,[]);compare(s,env.states)
                after=env.states[0].observation;cash=[z[0] for z in after.players]
                return dict(wallet=cash,margin=round(cash[1]-cash[0],2),scores=[k.reward for k in env.states],done=bool(env.done),
                            bank=[round(f.get("own_delivery",0),4) for f in ref.econ.cash_flow(s.cash.events)],events=copy.deepcopy(s.cash.events),
                            collisions=copy.deepcopy(s.last_collisions),state_sha256=fingerprint(physical(env.states)))
            baseline=settle(actions)
            check(physical(env.states)==physical(tape[t+1],True),"Baseline didn't reproduce")
            alts=[]
            for uid,data in {**players[1][2],**players[1][1]}.items():
                for a in ACTIONS if uid in players[1][2] else BASES:
                    if a==actions[ME].get(uid,"NONE"):continue
                    m=settle([actions[0],cmd(actions[1],uid,a)])
                    gain=round(m["margin"]-baseline["margin"],2)
                    collect=round(m["bank"][1]-baseline["bank"][1],4);deny=round(baseline["bank"][0]-m["bank"][0],4)
                    caught=[(event[0],event[2],event[3]) for event in m["collisions"] if event[1]==0 and event[4]==1 and event[2]>0 and event not in baseline["collisions"]]
                    score=round(m["scores"][1]-m["scores"][0],2) if m["done"] else None
                    alts.append(dict(uid=uid,original=actions[1].get(uid,"NONE"),move=a,player_before=data,delta_bank_me=collect,
                                     denied_bank_enemy=deny,new_enemy_cargo_seized=caught,margin_gain=gain,wallet=m["wallet"],terminal_margin=score))
            findings.append(dict(step=t,bank_by_seat=baseline["bank"],selected_margin=baseline["margin"],substitutions=len(alts),
                                 max_gain=max(x["margin_gain"] for x in alts),max_denial=max(x["denied_bank_enemy"] for x in alts),
                                 max_our_deposit_gain=max(x["delta_bank_me"] for x in alts),
                                 interception_options=[a for a in alts if a["new_enemy_cargo_seized"]],positive_options=[a for a in alts if a["margin_gain"]>0]))
            if t in (269,297):
                def two(uid,a1,a2,enemy_uid=None,enemy_choice=None):
                    ref.official_state.restore(env,frozen);s=copy.deepcopy(sim);stages=[]
                    queue=[actions, [k["action"] for k in tape[t+2]]]
                    for j,team in enumerate(queue):
                        same_orders=team[0] if j==0 or not enemy_uid else cmd(team[0],enemy_uid,enemy_choice)
                        selected=team[1] if (a1,a2)[j] is None else cmd(team[1],uid,(a1,a2)[j])
                        s.cash.events.clear();s.step(same_orders,selected);env.step([same_orders,selected],[]);compare(s,env.states)
                        o=env.states[0].observation
                        stages.append(dict(step=t+j+1,orders=[same_orders,selected],cash=[p[0] for p in o.players],margin=round(o.players[1][0]-o.players[0][0],2),
                                           own_worker=o.players[1][2].get(uid),enemy_worker=o.players[0][2].get(enemy_uid),collisions=copy.deepcopy(s.last_collisions),
                                           done=bool(env.done),scores=[k.reward for k in env.states]))
                    return stages
                if t==269:
                    uid="player-1-worker-72";eid="player-0-worker-66";recorded=tape[t+2][0]["action"].get(eid,"NONE")
                    intercept=dict(start=t,own=uid,enemy=eid,own_before=players[1][2][uid],enemy_before=players[0][2][eid],root_obs_sha256=fingerprint(physical(tape[t],True)),
                                   native_first=actions[1].get(uid,"NONE"),recorded_enemy_next=recorded,original=two(uid,None,None,eid,recorded),
                                   alternate_by_enemy_response={e:two(uid,"RIGHT","RIGHT",eid,e) for e in ACTIONS})
                else:
                    observed=round(game["rewards"][1]-game["rewards"][0],2);best=None;wins=0;ties=0;count=0
                    for uid in players[1][2]:
                        for a1 in ACTIONS:
                            for a2 in ACTIONS:
                                row=two(uid,a1,a2);count+=1
                                terminal=round(row[-1]["scores"][1]-row[-1]["scores"][0],2)
                                wins+=int(terminal>0);ties+=int(terminal==0)
                                if best is None or terminal>best["margin"]:
                                    best=dict(uid=uid,first=a1,second=a2,margin=terminal,stages=row)
                    final_two=dict(root=t,attempted_sequences=count,controls="one same worker two commands, public command to a worker no longer live is ignored by the rules; both enemy orders and nonselected own orders are historical",victories=wins,ties=ties,observed=observed,best=best)
            ref.official_state.restore(env,frozen)
        sim.cash.events.clear();sim.step(*actions);env.step(actions,[])
    check(physical(env.states)==physical(tape[299],True) and [v.reward for v in env.states]==game["rewards"] and env.done,"Wrong recorded ending")
    return dict(source=SOURCE.name,source_sha256=sha(raw),script_sha256=sha(Path(__file__).read_bytes()),seat=ME,terminal_official_scores_by_seat=game["rewards"],
                method="From each actual root t269..298 change exactly one legal collector/planter/base order; all other archived simultaneous commands are fixed. Each branch stops after one official transition; only t298 is an official terminal. Exhaustion includes commands whose recruitment is declined.",
                matched_historical_states=300,rows=findings,interception_two_step=intercept,last_two_steps=final_two)


if __name__=="__main__":
    results=investigate();out=HERE/"terminal_case.json"
    out.write_text(json.dumps(results,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(sum(x['substitutions'] for x in results['rows']))
    for x in results['rows']:
        print(x['step'],x['bank_by_seat'],x['max_gain'],x['max_denial'],x['max_our_deposit_gain'],[(a['uid'],a['move'],a['new_enemy_cargo_seized']) for a in x['interception_options']][:3])
