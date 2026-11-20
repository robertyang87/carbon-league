"""Through t267..279, test one collector deviation under the recorded simultaneous command."""
from pathlib import Path
import copy
import gzip
import hashlib
import importlib.util
import json
import random

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
SRC=HERE.parent/'v4_observations/increment_0904/replays/6e994d80674dc2430a4925b71bb6a947-1.json.gz'
spec=importlib.util.spec_from_file_location('rules',D/'freight_response/run.py')
rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
ACTIONS=['NONE','UP','RIGHT','DOWN','LEFT']
ME=0
def read():
    game=json.loads(gzip.decompress(SRC.read_bytes()))
    assert game['rewards']==[6857.57,7072.88]
    return game
def move(pos,k):
    step={'NONE':(0,0),'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    dy,dx=step[k];y,x=divmod(pos,15)
    return ((y+dy)%15)*15+(x+dx)%15
def dist(a,b):
    y,x=divmod(a,15);v,u=divmod(b,15);return min(abs(y-v),15-abs(y-v))+min(abs(x-u),15-abs(x-u))
def new_order(old,uid,cmd):
    result=dict(old)
    if cmd=='NONE':result.pop(uid,None)
    else:result[uid]=cmd
    return result
def after(sim, pair):
    s=copy.deepcopy(sim);s.cash.events.clear();s.step(*pair)
    flows=list(map(dict,rule.econ.cash_flow(s.cash.events)))
    return s,dict(cash=list(s.cash),margin=round(s.cash[0]-s.cash[1],2),
                  bank=[r.get('own_delivery',0) for r in flows],events=list(s.last_collisions),flows=flows)
def feasible(players,orders,who,a):
    w=players[0][2][who];goal=move(w[0],a)
    if any(uid!=who and move(info[0],orders.get(uid,'NONE'))==goal for uid,info in players[0][2].items()):return False
    if any(cell==goal and orders.get(uid,'NONE')!='NONE' for uid,cell in players[0][1].items()):return False
    return True
def own_capture(row):
    return {v[0]:v[2] for v in row['events'] if v[1]==1 and v[4]==0 and v[2]>0}
def act_history(fr,t,seat,uid,offsets):
    out=[]
    for k in sorted({max(0,min(298,t+i)) for i in offsets}):
        w=fr[k][0]['observation']['players'][seat][2].get(uid)
        out.append([k,w,(fr[k+1][seat]['action'] or {}).get(uid,'NONE') if w else None])
    return out
def prefilter():
    game=read();fr=game['steps'];sim=rule.econ.FastCarbon(game['configuration']).reset(full_carbon=fr[0][0]['observation']['full_carbon'])
    sim.cash=rule.econ.CashEvents(sim.cash);options=[];total=0
    for t in range(299):
        true=[row['action'] or {} for row in fr[t+1]]
        players=fr[t][0]['observation']['players']
        if t in range(267,280):
            base=after(sim,true)[1]
            for uid,w in players[0][2].items():
                if w[2]!='COLLECTOR':continue
                for a in ACTIONS:
                    if a==true[0].get(uid,'NONE') or not feasible(players,true[0],uid,a):continue
                    total+=1
                    pair=[new_order(true[0],uid,a),true[1]]
                    cand=after(sim,pair)[1]
                    cap=own_capture(cand);new={k:q for k,q in cap.items() if k not in own_capture(base)}
                    bank=cand['bank'][0]-base['bank'][0];deny=base['bank'][1]-cand['bank'][1]
                    if new or bank>1e-5 or deny>1e-5:
                        options.append(dict(step=t,uid=uid,from_state=w,before=true[0].get(uid,'NONE'),alt=a,new_enemy_cargo=new,
                                            actual_bank=base['bank'],alternate_bank=cand['bank'],bank_plus=bank,deny=deny,
                                            cash_gain=round(cand['margin']-base['margin'],2)))
        sim.cash.events.clear();sim.step(*true)
        assert sim.obs(0)['players']==fr[t+1][0]['observation']['players'] and list(sim.carbon)==fr[t+1][0]['observation']['full_carbon']
    return {'input_sha':hashlib.sha256(SRC.read_bytes()).hexdigest(),'single_collector_legal_joint_alternatives':total,'observed_same_step_positive':options}
if __name__=='__main__':
    s=prefilter();(HERE/'scan.json').write_text(json.dumps(s,indent=2)+'\n')
    for x in s['observed_same_step_positive']:print(x)
