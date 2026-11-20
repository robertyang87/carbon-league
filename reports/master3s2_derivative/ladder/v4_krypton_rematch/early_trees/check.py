"""One exposed tree substitution under historical commands and one simultaneous defense."""
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
LADDER=HERE.parents[1]
sys.path.insert(0,str(LADDER/'v4_alphacarbon_decisions'))
import branches as b

M='5906206b61c94c29d8ea2f99d2b4d62d'
TARGET='player-1-tree-4'
OTHER='player-1-tree-6'
PLANTER='player-0-worker-8'
GUARD='player-1-worker-6'
REG='0f2bb5eeeea6ecf9061a8ac45809b27f585c78f99ef6d825a7e5ae552b09a156'


def sha(x):
    return hashlib.sha256(x).hexdigest()


def where(ps,tree):
    return next(([s,*p[3][tree]] for s,p in enumerate(ps) if tree in p[3]),None)


def screen(fr,ledger):
    result=[]
    for item in ledger['planting'][1]['individual']:
        if (item['t'],item['cell']) not in {(19,117),(28,107),(31,59),(32,165)}:continue
        cell=item['cell'];near=[]
        for t in range(item['t']+1,item['end_t']+1):
            ps=fr[t][0]['observation']['players']
            mine=sorted((b.dist(v[0],cell),u,v[0]) for u,v in ps[0][2].items() if v[2]=='PLANTER')
            enemy=sorted((b.dist(v[0],cell),u,v) for u,v in ps[1][2].items())
            moment={}
            if mine and mine[0][0]==0:
                uid=mine[0][1];real=fr[t+1]
                moment['own_actual_command']=(real[0]['action'] or {}).get(uid,'NONE')
                moment['actual_enemy_entering']=[[u,pos,(real[1]['action'] or {}).get(u,'NONE'),kind]
                    for u,(pos,c,kind) in ps[1][2].items() if b.move(pos,(real[1]['action'] or {}).get(u,'NONE'))==cell]
            near.append(dict(step=t,wallet=ps[0][0],nearest_own_planters=mine,nearest_enemy=enemy[:2],**moment,
                             standing=bool(mine and mine[0][0]==0),affordable=ps[0][0]>=20))
        result.append(dict(actual=item,screens=near))
    assert len(result)==4
    return result


def run():
    protocol=(HERE/'protocol.json').read_bytes();assert sha(protocol)==REG
    d=LADDER/'v4_observations/increment_0926';sp=d/'replays'/f'{M}-1.json.gz';lp=d/f'{M}-1.ledger.json.gz'
    game=json.loads(gzip.decompress(sp.read_bytes()));f=game['steps'];cfg=game['configuration'];ledger=json.loads(gzip.decompress(lp.read_bytes()))
    assert sha(sp.read_bytes())==ledger['replay_sha256'] and ledger['seat']==0 and game['rewards']==[6245.53,6493.12] and cfg['seizeCost']==20
    exposures=screen(f,ledger)
    b.rule.np.random.seed(cfg['randomSeed']);random.seed(cfg['randomSeed'])
    env=b.rule.make('carbon',configuration=cfg);terrain=f[0][0]['observation']['full_carbon']
    env.states[0].observation.full_carbon=copy.deepcopy(terrain)
    sim=b.rule.econ.FastCarbon(cfg).reset(full_carbon=terrain);sim.cash=b.rule.econ.CashEvents(sim.cash);results=[]
    for t in range(299):
        assert b.view(env)==b.fast(sim)==b.archived(f[t])
        act=[v['action'] or {} for v in f[t+1]]
        if t==42:
            env.steps[:-1]=[None]*t;env.logs=[]
            snap=b.rule.official_state.take(env,[0,0])
            for defense in (False,True):
                for alter in (False,True):
                    b.rule.official_state.restore(env,snap);state=copy.deepcopy(sim);tape=[]
                    for k in range(42,82):
                        ps=state.obs(0)['players'];orig=[r['action'] or {} for r in f[k+1]]
                        ids=[set(p[1])|set(p[2]) for p in ps]
                        moves=[{u:a for u,a in m.items() if u in active} for m,active in zip(orig,ids)]
                        omitted=[sorted(set(m)-i) for m,i in zip(orig,ids)]
                        fresh=[sorted(set(p[2])-set(f[k][0]['observation']['players'][s][2])) for s,p in enumerate(ps)]
                        if alter and k==42:
                            assert ps[0][2][PLANTER]==[45,0,'PLANTER'] and b.feasible(ps,moves[0],PLANTER,'LEFT')
                            moves[0]=b.new_order(moves[0],PLANTER,'LEFT')
                        if alter and k==43 and PLANTER in ps[0][2] and ps[0][2][PLANTER][0]==59 and TARGET in ps[1][3] and ps[0][0]>=20 and b.feasible(ps,moves[0],PLANTER,'NONE'):
                            moves[0]=b.new_order(moves[0],PLANTER,'NONE')
                        if defense and k==42:
                            assert ps[1][2][GUARD][0]==44 and b.move(44,'DOWN')==59 and orig[1].get(GUARD,'NONE')=='NONE'
                            moves[1]=b.new_order(moves[1],GUARD,'DOWN')
                        state,flows=b.after(state,moves);env.step(moves,[])
                        actual=b.view(env);assert actual==b.fast(state)
                        if not alter and not defense:assert actual==b.archived(f[k+1])
                        after=state.obs(0)['players']
                        tape.append(dict(step=k,commands=moves,wallets=flows['cash'],margin=flows['margin'],collisions=flows['events'],
                                         planter=after[0][2].get(PLANTER),guard=after[1][2].get(GUARD),
                                         targets={v:where(after,v) for v in (TARGET,OTHER)},money=copy.deepcopy(state.cash.events),
                                         flows=flows['flows'],missing_archived_workers=omitted,new_workers_wait=fresh,
                                         referee_sha256=sha(json.dumps(actual,sort_keys=True,separators=(',',':')).encode())))
                    rent={tr:[sum(ev[2] for q in tape for ev in q['money'] if ev[0]=='tree_absorption' and ev[1]==s and ev[3]==tr) for s in (0,1)] for tr in (TARGET,OTHER)}
                    cash=[Counter(),Counter()]
                    for z in tape:
                        for s,v in enumerate(z['flows']):cash[s].update(v)
                    results.append(dict(defense=defense,alternate=alter,rent=rent,flows=cash,tape=tape))
            b.rule.official_state.restore(env,snap)
        env.step(act,[]);sim.cash.events.clear();sim.step(*act)
    assert b.fast(sim)==b.view(env)==b.archived(f[-1]) and [s.reward for s in env.states]==game['rewards']
    out=dict(protocol_sha256=REG,source_sha256=sha(sp.read_bytes()),ledger_sha256=sha(lp.read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),
             official_scores=game['rewards'],verified_steps=299,branches=results,targets=exposures,root_players=f[42][0]['observation']['players'])
    (HERE/'evidence.json.gz').write_bytes(gzip.compress((json.dumps(out,sort_keys=True,allow_nan=False)+'\n').encode(),mtime=0))
    for r in results:
        print('def/alt',r['defense'],r['alternate'],'rent',r['rent'],'snapshots',[(x['step']+1,x['wallets'],x['margin'],x['planter']) for x in r['tape'] if x['step'] in (42,43,49,59,81)])
        print('tenures',[(x['step'],x['targets']) for i,x in enumerate(r['tape']) if not i or x['targets']!=r['tape'][i-1]['targets'] and any((x['targets'][z] or [None])[0]!=(r['tape'][i-1]['targets'][z] or [None])[0] for z in (TARGET,OTHER))])
    for rr in exposures:
        ns=rr['screens'];print(rr['actual']['cell'],sum(v['standing'] for v in ns),[(v['step'],v['nearest_enemy'][0]) for v in ns if v['standing']])

if __name__=='__main__':run()
