"""Replay N33 cash, then change only one named worker against the historical orders."""
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
source_path = Path(sys.argv[1]) if len(sys.argv) == 2 else HERE.parents[1] / "replays/798a279b8c4cc1de4d5ab84de34d61ce-1.json.gz"
spec = importlib.util.spec_from_file_location("freight", HERE.parents[2] / "freight_response/run.py")
freight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freight)
raw = source_path.read_bytes()
game = json.loads(gzip.decompress(raw))
frames, cfg = game["steps"], game["configuration"]
assert game["rewards"] == [7289.39,7460.63] and len(frames) == 300 and cfg["size"] == 15
def roster(t):
    return frames[t][0]["observation"]["players"]
def orders(t):
    return [dict(row["action"] or {}) for row in frames[t+1]]
def distance(a,b):
    return sum(min((i-j)%15,(j-i)%15) for i,j in zip(divmod(a,15),divmod(b,15)))
moves = {"NONE":(0,0),"UP":(-1,0),"RIGHT":(0,1),"DOWN":(1,0),"LEFT":(0,-1)}
def dest(c,a):
    y,x=divmod(c,15); dy,dx=moves[a]
    return ((y+dy)%15)*15+(x+dx)%15
def fingerprint(s):
    return {k:s[k] for k in ("players","trees","step","full_carbon")}
def board(s,sim,t,original):
    assert fingerprint(s)==fingerprint(sim.obs(0) | {"full_carbon":list(sim.carbon)})
    if original:assert fingerprint(s)==fingerprint(frames[t][0]["observation"])
def watch(t, uid):
    w=roster(t)[int(uid.split("-")[1])][2].get(uid)
    return {"t":t,"cell":w[0],"load":w[1],"order":orders(t)[int(uid.split("-")[1])].get(uid,"NONE")} if w else {"t":t,"absent":True}
ds_70="player-0-worker-70"
def incoming(payee,frame):
    eligible={}
    for t in range(frame-10,frame):
        r=roster(t); prior=r[1][2].get(payee); pairs=[]
        if prior:
            for arrival in range(t+1,frame):
                w=roster(arrival)[1][2].get(payee)
                if w is None or w[0] in r[1][1].values():continue
                for uid,(cell,carry,role) in r[0][2].items():
                    d=distance(cell,w[0])
                    if role=="COLLECTOR" and carry<prior[1] and d<=arrival-t:
                        pairs.append(dict(uid=uid,cell=cell,cargo=carry,goal=w[0],arrival=arrival,steps=d,deadline=arrival-t))
        eligible[t]=pairs
    root=max(t for t,x in eligible.items() if x)
    return dict(frame=frame,payee=payee,latest_root=root,eligible=eligible,enemy_trajectory=[watch(t,payee) for t in range(frame-10,frame)])
tasks=[incoming("player-1-worker-55",290),incoming("player-1-worker-66",291)]
roots={task["latest_root"] for task in tasks}|{296}
random.seed(cfg["randomSeed"]);freight.np.random.seed(cfg["randomSeed"])
env=freight.make("carbon",configuration=cfg)
initial=frames[0][0]["observation"]["full_carbon"];env.states[0].observation.full_carbon=copy.deepcopy(initial)
sim=freight.econ.FastCarbon(cfg).reset(full_carbon=initial);sim.cash=freight.econ.CashEvents(sim.cash)
snapshots={};ledger=[];collisions=[]
for t in range(299):
    board(env.states[0].observation,sim,t,True)
    assert all(env.states[k].observation.carbon==frames[t][k]["observation"]["carbon"]==sim.obs(k)["carbon"] for k in (0,1))
    if t in roots:
        env.steps[:-1]=[None]*t;env.logs=[]
        snapshots[t]=(freight.official_state.take(env,[0.,0.]),copy.deepcopy(sim))
    joint=orders(t);sim.cash.events.clear();sim.step(*joint);env.step(joint,[])
    ledger.append(copy.deepcopy(sim.cash.events))
    collisions.append(sim.last_collisions[:])
board(env.states[0].observation,sim,299,True)
assert [s.reward for s in env.states]==game["rewards"]
def suffix(t,uid,path,end):
    saved,shadow=snapshots[t];freight.official_state.restore(env,saved);sim=copy.deepcopy(shadow);rows=[]
    for k in range(t,end):
        joint=orders(k);s=env.states[0].observation.players
        original=joint[0].get(uid,"NONE")
        if path is not None and k-t<len(path):
            choice=path[k-t]
            if choice=="NONE":joint[0].pop(uid,None)
            else:joint[0][uid]=choice
        sim.cash.events.clear();sim.step(*joint);env.step(joint,[])
        board(env.states[0].observation,sim,k+1,path is None)
        p=env.states[0].observation.players
        rows.append(dict(t=k,order=joint[0].get(uid,"NONE"),original_order=original,
            focal_before=s[0][2].get(uid),focal_after=p[0][2].get(uid),wallet=[x[0] for x in p],
            collision=sim.last_collisions[:],bank=[e for e in sim.cash.events if e[0]=="own_delivery"]))
    return dict(path=path,rows=rows,margin=round(p[0][0]-p[1][0],2),score=[st.reward for st in env.states] if env.done else None)
def minimal_walks(cell,goal,left,walk_count):
    if left==0:
        if cell==goal and walk_count==0: yield []
        return
    for cmd in moves:
        cost=int(cmd!="NONE");x=dest(cell,cmd)
        if 0<=walk_count-cost<=left-1 and distance(x,goal)<=walk_count-cost:
            for later in minimal_walks(x,goal,left-1,walk_count-cost):yield [cmd]+later
def snapshot_fates(trace,uid,payee):
    for row in trace["rows"]:
        if not row["focal_after"]:continue
        losses=[c for c in row["collision"] if c[0]==payee and c[3:5]==(0,0)]
        if losses:
            enemy=roster(row["t"])[1][2].get(payee)
            if enemy and row["focal_after"][0]==dest(enemy[0],orders(row["t"])[1].get(payee,"NONE")):return row["t"],losses[0][2]
    return None
for task in tasks:
    t=task["latest_root"];end=task["frame"]
    first=task["eligible"][t][0];task["control"]=suffix(t,first["uid"],None,end)
    task["routes"]=[]
    for test in task["eligible"][t]:
        for path in minimal_walks(test["cell"],test["goal"],test["deadline"],test["steps"]):
            uid=test["uid"];v=suffix(t,uid,path,end)
            task["routes"].append(dict(candidate=test,uid=uid,path=path,rows=v["rows"],focal_capture=snapshot_fates(v,uid,task["payee"]),end_margin=v["margin"],delta_vs_control=round(v["margin"]-task["control"]["margin"],2),actual_own_route=[watch(k,uid) for k in range(t,end)]))
    if task["payee"]=="player-1-worker-55":
        # At most 14 known actions: compare the registered four moves on the same
        # frozen tape through official settlement. Unlogged later orders are no-op.
        intact=suffix(t,first["uid"],None,299)
        assert intact["score"]==game["rewards"]
        for row in task["routes"]:
            final=suffix(t,row["uid"],row["path"],299)
            row["historical_frozen_suffix_to_terminal"]=final
            row["terminal_delta_vs_archived"]=round(final["margin"]-intact["margin"],2)
    task["actual_banked"]=[e for e in ledger[end-1] if e[0]=="own_delivery" and e[3]==task["payee"]]
    task["actual_collisions_deposit_step"]=collisions[end-1]
actual=suffix(296,ds_70,None,299)
variants=[suffix(296,ds_70,list(cmds),299) for cmds in itertools.product(moves,repeat=3)]
for v in variants:v['delta_margin']=round(v['margin']-actual['margin'],2)
def predanger(t,uid):
    p=roster(t);cell,cargo,_=p[0][2][uid];ret={}
    for cmd in moves:
        goal=dest(cell,cmd);might=[]
        for eid,(x,load,role) in p[1][2].items():
            if role=='COLLECTOR' and load<=cargo and distance(x,goal)<=1:
                eact=orders(t)[1].get(eid,'NONE')
                might.append(dict(uid=eid,cell=x,load=load,actual=eact,actually_hits=dest(x,eact)==goal))
        ret[cmd]=dict(cell=goal,d_home=min(distance(goal,b) for b in p[0][1].values()),possibly_lethal=might)
    return ret
groups={}
for cmd in moves:
    copies=[v for v in variants if v['path'][0]==cmd];best=max(copies,key=lambda x:x['margin']);groups[cmd]=dict(best=best['margin'],gain=best['delta_margin'],witness=best['path'],wins=sum(v['margin']>0 for v in copies))
def cash_components(s,a,b):
    c=Counter()
    for tick in ledger[a:b]:
        for kind,seat,change,*detail in tick:
            if seat==s:c[kind]+=change
    assert abs(sum(c.values())-roster(b)[s][0]+roster(a)[s][0])<1e-5
    return {k:round(v,5) for k,v in c.items()}
data=dict(match=source_path.name,sha256=hashlib.sha256(raw).hexdigest(),seat=0,actual_official_score=game['rewards'],
    scope="Postgame diagnostic. All original 300 observed states and official scores reproduced. Every intervention both engines checked. Only the given worker receives changed instructions. All other turns follow archived simultaneous commands, which omit newly unreachable entities in the rules. Future endpoint and routes are used in hindsight, not as legal decision-time inputs. One physical map, no adaptive opposing response and no prediction of win probability.",
    income={f"{a}_{b}":[cash_components(s,a,b) for s in (0,1)] for a,b in [(0,299),(270,299),(289,299),(294,299)]},
    stolen=[dict(t=t,id=uid,seat=s,amount=v,recipient=g,collector=winner) for t,cl in enumerate(collisions) for uid,s,v,g,winner in cl if g==1-s and v>0],
    intercepts=tasks,defend_296=dict(unit=ds_70,lookback=[watch(t,ds_70) for t in range(286,297)],visible_threats=predanger(296,ds_70),control=actual,by_first_action=groups,all_125_routes=variants))
dst=HERE/"cargo.json.gz";dst.write_bytes(gzip.compress((json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
print('defend',actual['margin'],groups)
for task in tasks:
    print('target',task['payee'],task['frame'],task['latest_root'],len(task['routes']),sum(bool(v['focal_capture']) for v in task['routes']),min(v['delta_vs_control'] for v in task['routes']),max(v['delta_vs_control'] for v in task['routes']))
