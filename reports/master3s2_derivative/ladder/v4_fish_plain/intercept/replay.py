"""An exposed ladder case: replay two fixed command paths under each of five simultaneous replies."""
import copy
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE.parents[1]
sys.path.insert(0,str(LADDER/'v4_alphacarbon_decisions'))
import branches as rules

M = 'bef37c25adeeca8f86eb6a52bcfab97d'
SRC = LADDER/'v4_observations/increment_0919/replays'/f'{M}-1.json.gz'
LOCK = 'e90ed107744951475a51d770f93084d3055f95dc04b91af34eea60bbf3e8ae19'
US = 'player-0-worker-13'
FOE = 'player-1-worker-14'
FOE15 = 'player-1-worker-15'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def corridor(ps, enemy, v):
    z = ps[1][2][enemy][0]
    homes = list(ps[1][1].values())
    delay = rules.dist(z,v)
    best = min(rules.dist(z,b) for b in homes)
    return v not in homes and 1<=delay<best and any(delay+rules.dist(v,b)==best for b in homes)


def screen(ps, target):
    pos,load,role = ps[1][2][target]
    bases = list(ps[1][1].values())
    options = []
    for cell in range(225):
        distance = rules.dist(pos,cell)
        if not 1<=distance<=4 or not corridor(ps,target,cell):
            continue
        for uid,(q,c,role) in ps[0][2].items():
            mine = rules.dist(q,cell)
            if role=='COLLECTOR' and c<load and mine<=distance:
                options.append([uid,cell,mine,distance])
    return dict(source=pos,carry=load,nearest_home=min(rules.dist(pos,b) for b in bases),
                intercept_cells=options)


def run():
    lock=(HERE/'protocol.json').read_bytes();assert sha(lock)==LOCK
    game=json.loads(gzip.decompress(SRC.read_bytes()));f=game['steps'];cfg=game['configuration']
    assert game['rewards']==[7903.,8273.56] and len(f)==300 and cfg['moveCost']==0
    root=142;cut=155
    assert len({case for case in rules.ACTIONS})==5
    geom=[{'t':t,'target14':screen(f[t][0]['observation']['players'],FOE),'target15':screen(f[t][0]['observation']['players'],FOE15)} for t in range(128,146)]
    random.seed(cfg['randomSeed']);rules.rule.np.random.seed(cfg['randomSeed'])
    env=rules.rule.make('carbon',configuration=cfg)
    terrain=f[0][0]['observation']['full_carbon']
    env.states[0].observation.full_carbon=copy.deepcopy(terrain)
    sim=rules.rule.econ.FastCarbon(cfg).reset(full_carbon=terrain)
    sim.cash=rules.rule.econ.CashEvents(sim.cash)
    cases=[]
    for t in range(299):
        assert rules.view(env)==rules.fast(sim)==rules.archived(f[t])
        archived_orders=[r['action'] or {} for r in f[t+1]]
        if t==root:
            env.steps[:-1]=[None]*t;env.logs=[]
            frozen=rules.rule.official_state.take(env,[0,0])
            for reply in rules.ACTIONS:
                for change in (False,True):
                    rules.rule.official_state.restore(env,frozen)
                    state=copy.deepcopy(sim);tape=[]
                    for k in range(root,cut):
                        p=state.obs(0)['players'];known=state.obs(0)['carbon']
                        original=[dict(r['action'] or {}) for r in f[k+1]]
                        present=[set(v[1])|set(v[2]) for v in p]
                        orders=[{u:a for u,a in o.items() if u in ids} for o,ids in zip(original,present)]
                        skipped=[sorted(set(o)-ids) for o,ids in zip(original,present)]
                        branch_extra=[sorted(set(p[i][2])-set(f[k][0]['observation']['players'][i][2])) for i in (0,1)]
                        if k==143:
                            assert FOE in p[1][2]
                            orders[1]=rules.new_order(orders[1],FOE,reply)
                        act=None
                        if change and US in p[0][2]:
                            pos,load,_=p[0][2][US]
                            if k==142:act='RIGHT'
                            elif k==143 and FOE in p[1][2] and rules.dist(pos,p[1][2][FOE][0])==1 and corridor(p,FOE,pos):act='NONE'
                            elif load>0:act=rules.hurry(p,US,orders[0])
                            if act and rules.feasible(p,orders[0],US,act):orders[0]=rules.new_order(orders[0],US,act)
                        state,flows=rules.after(state,orders)
                        env.step(orders,[])
                        assert rules.view(env)==rules.fast(state)
                        if not change and reply==(f[144][1]['action'] or {}).get(FOE,'NONE'):
                            assert rules.view(env)==rules.archived(f[k+1])
                        after=state.obs(0)['players']
                        events=copy.deepcopy(state.cash.events)
                        tape.append(dict(t=k,orders=orders,pre_worker=p[0][2].get(US),pre_target=p[1][2].get(FOE),
                                         worker=after[0][2].get(US),target=after[1][2].get(FOE),cash=flows['cash'],
                                         margin=flows['margin'],bank=flows['bank'],collisions=flows['events'],cash_events=events,
                                         missing_archived_agents=skipped,new_agents_wait=branch_extra,
                                         material_sha256=sha(json.dumps(rules.view(env),sort_keys=True,separators=(',',':')).encode())))
                    cases.append(dict(replacement=change,enemy_reply143=reply,tape=tape))
            rules.rule.official_state.restore(env,frozen)
        env.step(archived_orders,[]);sim.cash.events.clear();sim.step(*archived_orders)
    assert rules.fast(sim)==rules.archived(f[-1])==rules.view(env)
    assert [s.reward for s in env.states]==game['rewards']
    actual=(f[144][1]['action'] or {}).get(FOE,'NONE')
    out=dict(protocol_sha256=LOCK,replay_sha256=sha(SRC.read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),
             official_verified=299,match=M,official_scores=game['rewards'],screen=geom,actual_enemy_143=actual,
             actual_own_142=(f[143][0]['action'] or {}).get(US,'NONE'),
             start_players=f[142][0]['observation']['players'],
             before={str(t):[[f[t][0]['observation']['players'][s][2].get(u),(f[t+1][s]['action'] or {}).get(u,'NONE')] for s,u in [(0,US),(1,FOE),(1,FOE15)]] for t in (132,137,139,140,141,142)},cases=cases)
    (HERE/'branches.json.gz').write_bytes(gzip.compress((json.dumps(out,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode(),mtime=0))
    for a,b in zip(cases[::2],cases[1::2]):
        print(a['enemy_reply143'],[(c['t']+1,round(d['margin']-c['margin'],2), d['worker'],d['target']) for c,d in zip(a['tape'],b['tape']) if c['t'] in (143,145,154)])
    print('historical',actual,out['actual_own_142'])
    for q in geom:print(q['t'],len(q['target14']['intercept_cells']),len(q['target15']['intercept_cells']))

if __name__=='__main__':
    run()
