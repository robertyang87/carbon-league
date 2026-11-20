"""Official replay of two pre-bank interceptions into Ryan's actual t153 double deposit."""
import copy
import gzip
import hashlib
import importlib.util
import json
import random
from pathlib import Path

HERE=Path(__file__).resolve().parent
MATCH="1c80b916b203238a43267baeabc88fbe"
spec=importlib.util.spec_from_file_location("v4_official_checks",HERE.parent/"observations_next/fable_v7/terminal_case.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ref=old.ref

def dist(a,b):
    return sum(min((i-j)%15,(j-i)%15) for i,j in zip(divmod(a,15),divmod(b,15)))

def move(p,a):
    y,x=divmod(p,15); dx,dy={'LEFT':(-1,0),'RIGHT':(1,0),'UP':(0,-1),'DOWN':(0,1),'NONE':(0,0)}[a]
    return ((y+dy)%15)*15+(x+dx)%15

def route(players,uid):
    pos,load,kind=players[1][2][uid]
    bases=list(players[1][1].values()); enemy_bases=set(players[0][1].values())
    bad=[u[0] for u in players[0][2].values() if u[2]=='COLLECTOR' and u[1]<load]
    crowd={u[0] for k,u in players[1][2].items() if k!=uid}
    def priority(a):
        target=move(pos,a)
        return (target in enemy_bases, min((dist(target,z) for z in bad),default=15)<=1 and target not in bases,
                min(dist(target,b) for b in bases),target in crowd and target not in bases,old.ACTIONS.index(a))
    return min(old.ACTIONS,key=priority)

def study():
    path=HERE/'replays'/f'{MATCH}-1.json.gz';data=json.loads(gzip.decompress(path.read_bytes()));tape=data['steps'];cfg=data['configuration']
    random.seed(cfg['randomSeed']);ref.np.random.seed(cfg['randomSeed'])
    env=ref.make("carbon",configuration=cfg);env.states[0].observation.full_carbon=copy.deepcopy(tape[0][0]['observation']['full_carbon'])
    sim=ref.econ.FastCarbon(cfg).reset(full_carbon=env.states[0].observation.full_carbon);sim.cash=ref.econ.CashEvents(sim.cash)
    def check_phy():
        o=sim.obs(0)
        assert old.physical(env.states)==[sim.step_count,o['players'],o['trees'],list(sim.carbon),[sim.obs(0)['carbon'],sim.obs(1)['carbon']]]
    roots={}
    for t in range(299):
        assert old.physical(env.states)==old.physical(tape[t],True)
        if t in (149,150):
            env.steps[:-1]=[None]*t;env.logs=[]
            roots[t]=ref.official_state.take(env,[0,0]),copy.deepcopy(sim)
        acts=[p['action'] for p in tape[t+1]]
        env.step(acts,[]);sim.cash.events.clear();sim.step(*acts);check_phy()
    assert old.physical(env.states)==old.physical(tape[299],True)
    scenarios=[(150,16,21,['DOWN','RIGHT']), (150,16,21,['RIGHT','DOWN']), (150,14,24,['LEFT','LEFT']), (149,14,24,['LEFT','LEFT','DOWN'])]
    result=[]
    for t0,our,opp,orders in scenarios:
        worker=f'player-1-worker-{our}';enemy=f'player-0-worker-{opp}'
        def branch(mode,reply=None):
            nonlocal sim
            snap,template=roots[t0];ref.official_state.restore(env,snap);sim=copy.deepcopy(template);rows=[];delivered=[];events=[];follow=True
            for t in range(t0,166):
                vis=env.states[0].observation.players;our_units=vis[1][2];diff=vis[1][0]-vis[0][0]
                orig=[copy.deepcopy(p['action']) or {} for p in tape[t+1]]
                if mode!='actual' and worker in our_units:
                    if t-t0<len(orders):own=orders[t-t0]
                    elif mode=='escort' and follow:own=route(vis,worker)
                    else:own=orig[1].get(worker,'NONE')
                    orig[1]=old.cmd(orig[1],worker,own)
                if reply is not None and t==151 and enemy in vis[0][2]:orig[0]=old.cmd(orig[0],enemy,reply)
                before=copy.deepcopy([v[2].get(w) for v,w in [(vis[1],worker),(vis[0],enemy)]])
                sim.cash.events.clear();sim.step(*orig);env.step(orig,[]);check_phy()
                if mode=='actual' and reply is None:assert old.physical(env.states)==old.physical(tape[t+1],True)
                state=env.states[0].observation.players
                collisions=copy.deepcopy(sim.last_collisions); relevant=[v for v in collisions if str(worker) in str(v) or str(enemy) in str(v)]
                for event in sim.cash.events:
                    if event[0]=='own_delivery' and (worker in str(event) or enemy in str(event)):
                        delivered.append([t+1,event]);follow=follow and (worker not in str(event))
                record=dict(t=t,our_before=before[0],enemy_before=before[1],our_cmd=orig[1].get(worker,'NONE'),enemy_cmd=orig[0].get(enemy,'NONE'),
                            after=[state[1][2].get(worker),state[0][2].get(enemy)],margin=round(state[1][0]-state[0][0],2),
                            events=copy.deepcopy(sim.cash.events),contacts=relevant)
                if relevant:events.append(dict(t=t,contacts=relevant))
                rows.append(record)
            return dict(rows=rows,worker_specific_bank=delivered,collisions=events)
        baseline=branch('actual');history=branch('fixed');safe=branch('escort')
        replies={}
        for cmd in old.ACTIONS:
            if cmd==tape[152][0]['action'].get(enemy,'NONE'):continue
            a=branch('escort',cmd)
            replies[cmd]=dict(t152_our_enemy=a['rows'][151-t0]['after'],t153_margin=a['rows'][152-t0]['margin'],t166_margin=a['rows'][-1]['margin'])
        result.append(dict(t=t0,our=worker,enemy=enemy,orders=orders,actual=baseline,fixed=history,escort=safe,responses_at_t151=replies))
    path=HERE/'ryan_intercepts.json.gz'
    path.write_bytes(gzip.compress((json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode(),mtime=0))
    for q in result:
        t=q['t'];print(q['our'],q['orders'],[(k,q[k]['rows'][151-t]['after'],q[k]['rows'][152-t]['margin'],q[k]['rows'][-1]['margin'],q[k]['worker_specific_bank']) for k in ('actual','fixed','escort')],flush=True)

if __name__=='__main__':study()
