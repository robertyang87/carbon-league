"""Publicly defined one-turn interception screen and one official fixed-opponent transition."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
LADDER=HERE.parents[1]
DERIV=LADDER.parent
MATCH='ffb3d312918eb0c4d49a9912ce4f77f7'
GAME=LADDER/'nn_patch_cutoff/replays'/f'{MATCH}-1.json.gz'
SHA='b2db8b051c92cc174078fd45e28fabff5aeba604ea31305cf21e426d393a915b'
DS=('NONE','UP','RIGHT','DOWN','LEFT')
def dest(p,a):
 y,x=divmod(p,15);dy,dx=[(0,0),(-1,0),(0,1),(1,0),(0,-1)][DS.index(a)];return 15*((y+dy)%15)+(x+dx)%15
def dist(a,b):
 y,x=divmod(a,15);v,u=divmod(b,15);return min(abs(y-v),15-abs(y-v))+min(abs(x-u),15-abs(x-u))
def choose(players,own):
 bases=set(players[1][1].values()); crew=players[0][2];planned={u:dest(w[0],own.get(u,'NONE')) for u,w in crew.items()}
 ordered=[]
 for enemy,(p,load,role) in players[1][2].items():
  if role!='COLLECTOR' or load<100 or min(dist(p,h) for h in bases)!=2:continue
  close=sorted({dest(p,a) for a in DS if min(dist(dest(p,a),h) for h in bases)==1})
  if len(close)!=1:continue
  cell=close[0]
  if cell in planned.values():continue
  for uid,(op,mine,kind) in crew.items():
   if kind!='COLLECTOR' or mine>25: continue
   for a in DS:
    if dest(op,a)==cell:ordered.append((-load,mine,uid,enemy,a,p,op,cell))
 return min(ordered) if ordered else None
def state(frame):
 o=frame[0]['observation']
 return [o['step'],o['players'],o['trees'],list(o['full_carbon']),[f['observation']['carbon'] for f in frame]]
def main():
 assert hashlib.sha256(GAME.read_bytes()).hexdigest()==SHA
 f=json.loads(gzip.decompress(GAME.read_bytes()));frames=f['steps']; cfg=f['configuration']
 chosen=next(((t,r) for t in range(50,251) if (r:=choose(frames[t][0]['observation']['players'],frames[t+1][0]['action'] or {}))),None)
 report=dict(match=MATCH,source_sha256=SHA,selection_rule='Earliest t=50..250; enemy collector load>=100, distance 2 to closest base, unique one-step return route; own collector load<=25 could occupy that non-base square in one move, none of own issued orders already takes it. Tie: descending enemy cargo, ascending own cargo then unit IDs. Screen sees only the pre-action shared roster and the acting side own anchor commands. Retrospective physical feasibility; no alternative selected on post-root result.')
 if not chosen:
  report['selected']=None
 else:
  t,r=chosen;neg,cargo,uid,victim,move,enemy_pos,my_pos,target=r
  report.update(selected=dict(t=t,uid=uid,victim=victim,cargo=cargo,enemy_cargo=-neg,my_pos=my_pos,enemy_pos=enemy_pos,target=target,requested=move))
  spec=importlib.util.spec_from_file_location('ryan_branch_harness',DERIV/'freight_response/run.py')
  h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
  def load():
   h.np.random.seed(cfg['randomSeed']); env=h.make('carbon',configuration=cfg);board=frames[0][0]['observation']['full_carbon'];env.states[0].observation.full_carbon=copy.deepcopy(board)
   sim=h.econ.FastCarbon(cfg).reset(full_carbon=board);sim.cash=h.econ.CashEvents(sim.cash)
   for j in range(t):
    orders=[x['action'] or {} for x in frames[j+1]];env.step(orders,[]);sim.step(*orders)
   assert state(env.states)==state(frames[t]);return env,sim
  for arm in ('actual','intercept'):
   env,sim=load();order=[copy.deepcopy(x['action'] or {}) for x in frames[t+1]];before=order[0].get(uid,'NONE')
   if arm=='intercept':
    if move=='NONE':order[0].pop(uid,None)
    else:order[0][uid]=move
   sim.cash.events.clear();env.step(order,[]);sim.step(*order);post=env.states[0].observation.players
   assert sim.obs(0)['players']==post and list(sim.carbon)==env.states[0].observation.full_carbon
   if arm=='actual':assert state(env.states)==state(frames[t+1])
   result=dict(actual_own=before,enemy_action=order[1].get(victim,'NONE'),enemy_destination=dest(enemy_pos,order[1].get(victim,'NONE')),
     our_after=post[0][2].get(uid),enemy_after=post[1][2].get(victim),cash=[team[0] for team in post],cash_events=sim.cash.events,collision_events=sim.last_collisions)
   report[arm]=result
 result=HERE/'probe.json';result.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 print(report['selected'],[(k,report[k]['enemy_action'],report[k]['our_after'],report[k]['enemy_after'],report[k]['cash']) for k in ('actual','intercept') if k in report])
if __name__=='__main__':main()
