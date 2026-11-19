"""Re-evaluate real final actions and conditional return with the unchanged official opponent."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"conditional"))
import responses as r

r.HERE=HERE
r.PLAN=HERE/"protocol.json"
r.P=r.load(r.PLAN)
r.PACK["candidate"]=HERE/"candidate.py"


def collisions(sim,v,actions):
    _,bases,workers,_=v["players"][v["player"]]
    destinations={}
    for id,(pos,_,_) in workers.items():
        act=actions.get(id)
        destinations[id]=sim._move_nbr[act][pos] if act in sim._move_nbr else pos
    destinations.update({id:pos for id,pos in bases.items() if actions.get(id)})
    return {tuple(sorted((a,b))) for a,i in destinations.items() for b,j in destinations.items() if a!=b and i==j}


def choose(case):
    raw=r.inputs(case);seat=case["seat"]
    models=[r.live.module(path,"collector_gate_"+name) for name,path in r.PACK.items()]
    sim=r.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    assert list(sim.carbon)==raw["initial_full_carbon"]
    proof=[]; selected=None
    for t,joint in enumerate(raw["actions"]):
        view=sim.obs(seat)
        if t in range(0,241,30) or 250<=t<=278:
            before,after=[r.live.clean(m.agent(view,raw["game_configuration"])) for m in models]
            assert before==r.live.clean(joint[seat]),"actual parent differs"
            bases=view["players"][seat][1];workers=view["players"][seat][2]
            changed={id:dict(role=("base" if id in bases else workers[id][2]),before=before.get(id,"NONE"),after=after.get(id,"NONE"))
                 for id in set(before)|set(after) if before.get(id)!=after.get(id)}
            new=collisions(sim,view,after)-collisions(sim,view,before)
            assert not new and all(a["role"]=="COLLECTOR" for a in changed.values()),"protected role or friendly collision changed"
            proof.append(dict(t=t,units=len(workers),bases=len(bases),planters=sum(w[2]=="PLANTER" for w in workers.values()),changed=changed))
            if selected is None and t>=250 and changed:
                selected=dict(**case,t=t,parent=before,candidate=after,opponent=joint[1-seat])
        else:
            for m in models:m._FEAT.observe(view)
        sim.step(*joint)
        if t>=278: break
    return dict(root=selected,observed=proof)


def run():
    audits=[choose(c) for c in r.P["exposures"]]
    r.write(HERE/"mechanism.json",dict(protocol=r.sha(r.PLAN),cases=audits))
    if not all(a["root"] for a in audits):
        r.write(HERE/"gate.json",dict(eligible=False,reason="no legal collector action change in fixed window"))
        return
    rows=[]
    with r.LOCK.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for audit in audits:
            c=audit["root"];seat=c["seat"]
            path=HERE/f"shared_warm_{c['seed']}_{seat}.json"
            if not path.exists():
                with ProcessPoolExecutor(1) as pool:cost=pool.submit(r.native_warm).result()
                r.write(path,dict(cost=cost,protocol=r.sha(r.PLAN)))
            shared=r.load(path)
            assert shared["protocol"]==r.sha(r.PLAN)
            with ProcessPoolExecutor(2) as pool:
                parent,candidate=pool.map(r.arm,[(c,"parent",seat,shared["cost"]),(c,"candidate",1-seat,shared["cost"])])
            assert parent["default_warm_cost"]==candidate["default_warm_cost"]==shared["cost"] and parent["initial_rng"]==candidate["initial_rng"]
            diff=[i+c["t"] for i,(p,q) in enumerate(zip(parent["tape"],candidate["tape"])) if p[1-seat]!=q[1-seat]]
            row=dict(seed=c["seed"],seat=seat,t=c["t"],changed_enemy_steps=diff,original_calls=[len(v["calls"]) for v in (parent,candidate)],
                margins=[v["margin"] for v in (parent,candidate)],delta=candidate["margin"]-parent["margin"],
                delta_tree=candidate["cash"]["tree_income_margin"]-parent["cash"]["tree_income_margin"],
                delta_delivery=candidate["cash"]["delivery_margin"]-parent["cash"]["delivery_margin"])
            rows.append(row)
            print(c["seed"],c["t"],row["margins"],row["delta"],len(diff),flush=True)
    r.write(HERE/"gate.json",dict(protocol=r.sha(r.PLAN),eligible=all(v["delta"]>0 and v["changed_enemy_steps"] for v in rows),rows=rows))


if __name__=="__main__":run()
