"""Reconstruct historical tapes, exercise actual prefix edits and independently audit the four new continuations."""
from collections import Counter
from copy import deepcopy
import argparse
import hashlib
from pathlib import Path
import sys

from run import ACTOR, HERE, PLAN, PARENT, R, START, END, Intervention, archive, audit, identity, np

DIRECTIONS=("NONE","UP","RIGHT","DOWN","LEFT")
DELTA={"NONE":(0,0),"UP":(-1,0),"RIGHT":(0,1),"DOWN":(1,0),"LEFT":(0,-1)}


def collision(view, command):
    d={}
    _,bases,workers,_=view["players"][view["player"]]
    for uid,(cell,_,_) in workers.items():
        dr,dc=DELTA[command.get(uid,"NONE")]
        d[uid]=((cell//15+dr)%15)*15+(cell%15+dc)%15
    for uid,cell in bases.items():
        if uid in command:d[uid]=cell
    return {(a,b) for a in d for b in d if a<b and d[a]==d[b]}


def fixture(case):
    raw=archive(case)
    m=R.live.module(PARENT,"fixture_incumbent")
    ed=Intervention(m)
    env=R.live.begin(raw)
    s=case["seat"]
    examples=[]
    for t,joint in enumerate(raw["actions"]):
        v=R.live.public(env,s)
        if t==0:m.agent(v,raw["game_configuration"])
        elif t in (START,START+4,START+8,START+12):
            # Repeat the identical observed trajectory prefix for each physical intervention.
            old=deepcopy(m._FEAT)
            normal=R.live.clean(m.agent(v,raw["game_configuration"]))
            for u,(_,_,kind) in v["players"][s][2].items():
                if kind!="PLANTER":continue
                for cmd in DIRECTIONS:
                    m._FEAT=deepcopy(old)
                    example=dict(normal)
                    example.pop(u,None)
                    if cmd!="NONE":example[u]=cmd
                    base,edited,row=ed.select(v,raw["game_configuration"],example)
                    assert base==normal and row["proposed"].get(u,"NONE")==cmd
                    projected=ed.model._FEAT.labels(row["proposed"])
                    physical=ed.model._FEAT.to_actions(projected)
                    assert physical==row["proposed"] and all(np.array(physical.get(i,"NONE")==example.get(i,"NONE"))
                        for i,w in v["players"][s][2].items() if w[2]=="PLANTER")
                    assert not collision(v,edited)-collision(v,base)
                    assert row["rejected"] or edited==row["proposed"]
                    if not row["rejected"] and base.get(u)!=edited.get(u):
                        for before in m._FEAT.own_uid:
                            if before==u:break
                            if before is not None:assert base.get(before)==edited.get(before)
                    examples.append(dict(t=t,uid=u,cmd=cmd,physical_change=base.get(u)!=edited.get(u),rejected=row["rejected"]))
                break
            m._FEAT=old
            m._FEAT.observe(v)
        else: m._FEAT.observe(v)
        env.step(joint)
    actual=[st.reward for st in env.states]
    assert actual==raw["official_score_by_seat"]
    cash=[p[0] for p in env.states[0].observation.players]
    audit(raw,raw["actions"][START:],cash)
    assert any(x["physical_change"] for x in examples)
    return dict(seed=case["seed"],seat=s,examples=examples,replayed_archived_scores=actual)


def verify():
    rows=[]
    for case in PLAN["cases"]:
        raw=archive(case);s=case["seat"]
        arms={}
        for label in ("parent","teacher"):
            file=HERE/f"{case['seed']}_{s}_{label}.json.gz"
            data=R.load(file)
            assert data["binding"]["program"]==identity()["program"]
            env=R.live.begin(raw)
            for both in raw["actions"][:START]:env.step(both)
            changed=0
            for i,both in enumerate(data["tape"]):
                if i<END-START:
                    r=data["decisions"][i]
                    assert r["actual"]==both[s]
                    assert (r["eligible"] if label=="teacher" else r["parent"])==r["actual"]
                    view=R.live.public(env,s)
                    assert not collision(view,r["actual"])-collision(view,r["parent"])
                    if label=="teacher":
                        changed+=int(any(r["actual"].get(u,"NONE")!=r["parent"].get(u,"NONE") for u,v in view["players"][s][2].items() if v[2]=="PLANTER"))
                env.step(both)
            official=[st.reward for st in env.states]
            assert official==data["scores"] and [st.status for st in env.states]==["DONE","DONE"]
            assert len(data["enemy_calls"])==len(data["tape"]) and [x["step"] for x in data["enemy_calls"]]==list(range(START,START+len(data["tape"])))
            assert [x["step"] for x in data["teacher_calls"]]==list(range(START,END))
            arms[label]=dict(sha=R.sha(file),scores=official,real_changed_planter_turns=changed,enemy_wall=sum(r["public_wall_seconds"] for r in data["enemy_calls"]))
        pair=R.load(HERE/f"pair_{case['seed']}.json")
        actual=(arms["teacher"]["scores"][s]-arms["teacher"]["scores"][1-s])-(arms["parent"]["scores"][s]-arms["parent"]["scores"][1-s])
        assert actual==pair["delta_margin"] and pair["changed_planter_steps"]==arms["teacher"]["real_changed_planter_turns"]
        rows.append(dict(**pair,arms=arms))
    qualifies=all(r["changed_planter_steps"]>=2 and r["delta_margin"]>0 for r in rows) and sum(r["cash_delta_by_horizon"]["to199"].get("ground_rent",0) for r in rows)>0
    R.write(HERE/"results.json",dict(protocol=identity()["protocol"],verifier=R.sha(Path(__file__)),rows=rows,teacher_data_signal_met=qualifies))
    print([(r["seed"],r["delta_margin"],r["changed_planter_steps"]) for r in rows],qualifies)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("fixture","verify"));args=p.parse_args()
    if args.stage=="verify":verify()
    else:
        output=[fixture(c) for c in PLAN["cases"]]
        R.write(HERE/"controlled_edits.json",dict(identity=identity(),verifier=R.sha(Path(__file__)),cases=output))
        print([(r["seed"],len(r["examples"]),sum(x["physical_change"] for x in r["examples"])) for r in output])
