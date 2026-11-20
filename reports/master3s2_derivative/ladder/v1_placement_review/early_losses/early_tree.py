"""Explore the first unrealized young-tree seizure in two old defeats and one same-opponent win."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
LADDER = HERE.parents[1]
SOURCE = LADDER.parent / "freight_response/run.py"
spec = importlib.util.spec_from_file_location("economic_referee", SOURCE)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

MATCHES = {
    "160624b76d6278dfb1e6ae1d145edce0": ("Makabaka-v1.3", -27.57),
    "3f0a1dfa0e1ac4a00a9f63f9cf644af9": ("哈基米v3pro", -272.66),
    "cd651c5cbf982d68df4fe2f5633398d6": ("哈基米v3pro", 1496.84),
}
VECTORS = {"NONE":(0,0),"UP":(-1,0),"RIGHT":(0,1),"DOWN":(1,0),"LEFT":(0,-1)}


def dest(cell, command):
    y,x = divmod(cell,15)
    dy,dx = VECTORS[command]
    return ((y+dy)%15)*15+(x+dx)%15


def tree_owner(players,tree):
    return next((s for s,p in enumerate(players) if tree in p[3]), None)


def select(frames, fee):
    choices = []
    for t in range(1,100):
        obs=frames[t][0]["observation"];p=obs["players"]
        targeted={v[0]:(tid,v[1]) for tid,v in p[0][3].items()}
        commands=frames[t+1][1]["action"] or {}
        for uid,(pos,cargo,kind) in p[1][2].items():
            if kind!="PLANTER" or pos not in targeted or p[1][0]<fee or commands.get(uid,"NONE")=="NONE":
                continue
            tid,age=targeted[pos];last_income=obs["trees"][tid][1]
            if age>20 or last_income<3:continue
            guards=[]
            for enemy,(location,load,job) in p[0][2].items():
                covers=[cmd for cmd in VECTORS if dest(location,cmd)==pos]
                if covers:guards.append(dict(uid=enemy,cell=location,cargo=load,kind=job,to_tree=covers[0]))
            choices.append(dict(t=t,tree=tid,age=age,last_receipt=last_income,seat1_cash=p[1][0],cell=pos,uid=uid,logged=commands[uid],guards=guards))
    return choices


def one(mid, opponent, actual_margin):
    path=LADDER/"replays"/f"{mid}-1.json.gz"
    raw=path.read_bytes();g=json.loads(gzip.decompress(raw));f=g["steps"];cfg=g["configuration"]
    assert round(g["rewards"][1]-g["rewards"][0],2)==actual_margin
    tickets=select(f,cfg["seizeCost"]);ticket=tickets[0];t=ticket["t"];tend=t+20
    guards=ticket["guards"];assert len(guards)==1
    guard=guards[0];tree=ticket["tree"];worker=ticket["uid"]
    def orders(k):return [dict(p.get("action") or {}) for p in f[k+1]]
    def check(env,shadow,k,logged):
        o=env.states[0].observation;sp=shadow.obs(0)
        assert o.step==k and o.players==sp["players"] and o.trees==sp["trees"] and o.full_carbon==list(shadow.carbon)
        assert [env.states[s].observation.carbon for s in (0,1)]==[shadow.obs(s)["carbon"] for s in (0,1)]
        if logged:
            old=f[k][0]["observation"]
            assert all(o[q]==old[q] for q in ("step","players","trees","full_carbon"))
    random.seed(cfg["randomSeed"]);ref.np.random.seed(cfg["randomSeed"])
    env=ref.make("carbon",configuration=cfg);initial=f[0][0]["observation"]["full_carbon"]
    env.states[0].observation.full_carbon=copy.deepcopy(initial)
    sim=ref.econ.FastCarbon(cfg).reset(full_carbon=initial);sim.cash=ref.econ.CashEvents(sim.cash)
    for k in range(t):
        check(env,sim,k,True)
        a=orders(k);sim.step(*a);env.step(a,[])
    check(env,sim,t,True);env.steps[:-1]=[None]*t;env.logs=[]
    frozen=ref.official_state.take(env,[0,0]);arms=[]
    for seize,protect in ((False,False),(True,False),(False,True),(True,True)):
        ref.official_state.restore(env,frozen);shadow=copy.deepcopy(sim);steps=[]
        for k in range(t,tend):
            joint=orders(k)
            if k==t:
                if seize:joint[1].pop(worker,None)
                if protect:
                    command=guard["to_tree"]
                    if command=="NONE":joint[0].pop(guard["uid"],None)
                    else:joint[0][guard["uid"]]=command
            shadow.cash.events.clear();shadow.step(*joint);env.step(joint,[]);check(env,shadow,k+1,not seize and not protect)
            players=env.states[0].observation.players
            steps.append(dict(t=k,orders=joint,focal_planter=players[1][2].get(worker),focal_guard=players[0][2].get(guard["uid"]),
                              tree_owner=tree_owner(players,tree),treasuries=[p[0] for p in players],margin=round(players[1][0]-players[0][0],2),
                              tree_absorption=[e for e in shadow.cash.events if e[0]=="tree_absorption" and e[3]==tree],
                              cash_flow=[dict(v) for v in ref.econ.cash_flow(shadow.cash.events)],
                              collision=shadow.last_collisions[:]))
        arms.append(dict(seize=seize,protect=protect,steps=steps))
    history=[]
    for k in range(max(0,t-10),t+1):
        p=f[k][0]["observation"]["players"];joint=orders(k)
        history.append(dict(t=k,planter=p[1][2].get(worker),guard=p[0][2].get(guard["uid"]),
                            planter_action=joint[1].get(worker,"NONE"),guard_action=joint[0].get(guard["uid"],"NONE"),
                            tree_owner=tree_owner(p,tree)))
    # Cash intervals refer to all actually recorded events, and do not imply the
    # alternative root action would change the observed whole-game result.
    phases={}
    for lo,hi in [(0,25),(25,50),(50,75),(75,100)]:
        def wallet(k):return [p[0] for p in f[k][0]["observation"]["players"]]
        left,right=wallet(lo),wallet(hi)
        phases[str(lo)]=dict(lo=lo,hi=hi,left=left,right=right,margin_change=round((right[1]-right[0])-(left[1]-left[0]),2))
    return dict(id=mid,src_sha256=hashlib.sha256(raw).hexdigest(),opponent=opponent,ds_seat=1,
                observed_scores=g["rewards"],actual_margin=actual_margin,phases=phases,lookback=history,
                matching_early_states=tickets,selected=ticket,branches=arms)


def main():
    all_cases=[one(mid,*info) for mid,info in MATCHES.items()]
    data=dict(scope="Exposed historical maps. First opportunity in actions t1..99 where the logged planter leaves an enemy tree aged at most 20, incoming public tree receipt at least 3, own treasury sufficient for seizure. One root-only alternative stays on the tree. Compare both root choices under the logged opposing order and the adjacent opponent going to the tree; later joint orders are those recorded. Holds for 20 transitions, with full physical board agreement in two engines, no opponent model or whole-game strength claim. Positive and negative examples retained.",cases=all_cases)
    (HERE/"branches.json.gz").write_bytes(gzip.compress((json.dumps(data,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
    for c in all_cases:
        print(c["id"][:8],c["selected"],[(a["seize"],a["protect"],[(x["tree_owner"],x["margin"]) for x in [a["steps"][i-1] for i in (1,5,10,20)]]) for a in c["branches"]])
if __name__=="__main__":
    main()
