"""Read opponent commands only after candidate freeze. One-step and short fixed-response checks are not matches."""
import copy
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np

import checks

HERE=Path(__file__).resolve().parent
F=checks.D/"freight_response/run.py"
import importlib.util
s=importlib.util.spec_from_file_location("archived_referee",F)
ref=importlib.util.module_from_spec(s);s.loader.exec_module(ref)


def replace(team,uid,a):
    row=team.copy()
    if a=="NONE":row.pop(uid,None)
    else:row[uid]=a
    return row


def settle():
    doc=json.loads(gzip.decompress((HERE/"checks.json.gz").read_bytes()))
    result=[]
    for row in doc["rows"]:
        if row['name'] not in ("Fable v7","doubao placement"):continue
        gid=row['id'];start=row['step'];me=row['seat'];foe=1-me
        path=checks.D/"ladder/replays"/(gid+"-1.json.gz")
        if not path.exists():path=checks.D/"ladder/placement_refresh/replays"/path.name
        data=json.loads(gzip.decompress(path.read_bytes()));tape=data['steps'];cfg=data['configuration']
        random.seed(cfg['randomSeed']);np.random.seed(cfg['randomSeed']);env=ref.make('carbon',configuration=cfg)
        terrain=tape[0][0]['observation']['full_carbon'];env.states[0].observation.full_carbon=copy.deepcopy(terrain)
        sim=ref.econ.FastCarbon(cfg).reset(full_carbon=terrain)
        def verify(r):
            public=env.states[0].observation
            assert r.obs(0)['players']==public.players and r.obs(0)['trees']==public.trees and list(r.carbon)==public.full_carbon
        for t in range(start):
            actual=[k['action'] or {} for k in tape[t+1]]
            env.step(actual,[]);sim.step(*actual);verify(sim)
            assert sim.obs(0)['players']==tape[t+1][0]['observation']['players'] and list(sim.carbon)==tape[t+1][0]['observation']['full_carbon']
        env.steps[:-1]=[None]*start;env.logs=[];snap=ref.official_state.take(env,[0,0]);sim.cash=ref.econ.CashEvents(sim.cash)
        v2,v3=row['runs'][:2]; seen=tape[start+1][me]['action'] or {};enemy_root=tape[start+1][foe]['action'] or {}
        who,enemy=checks.FOCAL[row['name']]
        if start==218:enemy="player-1-worker-52"
        controls={"archived":seen,"v2_issued":v2['issued'],"v3_issued":v3['issued']}
        if v3['audit']:
            rec=v3['audit'][0]
            controls.update(recorded_v3_A0=rec['original'],v3_proposal=rec['added'])
        if start==218:controls["wrong_heavier_pursuit"]=replace(seen,who,"RIGHT")
        def arm(ours,opp,stop):
            ref.official_state.restore(env,snap);r=copy.deepcopy(sim);turns=[]
            for t in range(start,stop):
                actual=[dict(k['action'] or {}) for k in tape[t+1]]
                if t==start:actual[me]=ours;actual[foe]=opp
                r.cash.events.clear();env.step(actual,[]);r.step(*actual);verify(r);ps=r.obs(0)['players']
                if ours==seen and opp==enemy_root:
                    assert r.obs(0)['players']==tape[t+1][0]['observation']['players'] and list(r.carbon)==tape[t+1][0]['observation']['full_carbon']
                turns.append(dict(t=t+1,our=ps[me][2].get(who),enemy=ps[foe][2].get(enemy),cash=list(r.cash),
                                  margin=round(r.cash[me]-r.cash[foe],2),collisions=copy.deepcopy(r.last_collisions),
                                  deposits=[e for e in r.cash.events if e[0]=="own_delivery"]))
            return turns
        arms={k:arm(v,enemy_root,start+6) for k,v in controls.items()}
        matrix={}
        if v3['audit']:
            for move in ("NONE","UP","RIGHT","DOWN","LEFT"):
                response=replace(enemy_root,enemy,move)
                matrix[move]={k:arm(controls[k],response,start+1)[0] for k in ("recorded_v3_A0","v3_proposal")}
        result.append(dict(id=gid,t=start,seat=me,focal=who,enemy=enemy,observed_enemy_step=enemy_root.get(enemy,"NONE"),
                           historical_opponent_arms=arms,mutually_legal_root_enemy_moves=matrix))
        print(start,[(k,r[0]['our'],r[0]['enemy'],r[-1]['margin']) for k,r in arms.items()],flush=True)
    checks.save(HERE/'cash_check.json.gz',dict(candidate_sha256=doc['candidate'],rows=result,method="Historical opposition at root and all subsequent tabled commands. Conditional six-step response; alternative enemy response matrix only advances one step. Both engines compared after every transition."))
if __name__=="__main__":settle()
