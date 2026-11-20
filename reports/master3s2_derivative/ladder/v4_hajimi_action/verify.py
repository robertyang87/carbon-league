"""Reproduce the benchmark loss and branch the two public actions protecting cell 27."""
import itertools
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v4_alphacarbon_decisions'))
from branches import rule,copy,gzip,json,hashlib,random,archived,view,fast,after,feasible,dist,move,new_order,act_history,ACTIONS
SOURCE=HERE.parent/'v4_hajimi_benchmark/replays/687ecf4fee697c81406fe6bbc4735141-1.json.gz'
OWN='player-0-worker-2';FOE='player-1-worker-11';GUARD='player-1-worker-10'
def active(p,seat,uid):
    return uid in p[seat][2]
def run():
    raw=SOURCE.read_bytes(); game=json.loads(gzip.decompress(raw));f=game['steps'];cfg=game['configuration']
    assert game['rewards']==[8174.49,9172.02] and game['statuses']==['DONE','DONE']
    random.seed(cfg['randomSeed']);rule.np.random.seed(cfg['randomSeed'])
    env=rule.make('carbon',configuration=cfg);first=f[0][0]['observation']['full_carbon']
    env.states[0].observation.full_carbon=copy.deepcopy(first)
    sim=rule.econ.FastCarbon(cfg).reset(full_carbon=first);sim.cash=rule.econ.CashEvents(sim.cash)
    roots=[];states=[]
    for start in range(299):
        true=[q['action'] or {} for q in f[start+1]]
        assert view(env)==fast(sim)==archived(f[start])
        if start==41:
            env.steps[:-1]=[None]*start;env.logs=[]
            frozen=rule.official_state.take(env,[0,0])
            response_grid=[('recorded','recorded',a) for a in ACTIONS]
            response_grid += [(a,b,'NONE') for a,b in itertools.product(ACTIONS,repeat=2) if (a,b)!=('RIGHT','RIGHT')]
            for enemy11,enemy10_now,enemy10 in response_grid:
                for intervention in (False,True):
                    rule.official_state.restore(env,frozen)
                    local=copy.deepcopy(sim);trace=[]
                    for t in range(start,48):
                        p=local.obs(0)['players']; archive=[q['action'] or {} for q in f[t+1]]
                        pools=[set(x[1])|set(x[2]) for x in p]
                        pair=[{u:a for u,a in o.items() if u in pool} for o,pool in zip(archive,pools)]
                        absent=[sorted(set(o)-pool) for o,pool in zip(archive,pools)]
                        if t==41 and enemy11!='recorded' and active(p,1,FOE):
                            pair[1]=new_order(pair[1],FOE,enemy11)
                        if t==41 and enemy10_now!='recorded' and active(p,1,GUARD):
                            pair[1]=new_order(pair[1],GUARD,enemy10_now)
                        if t==42 and active(p,1,GUARD):
                            pair[1]=new_order(pair[1],GUARD,enemy10)
                        if intervention and t in (41,42) and active(p,0,OWN):
                            a={41:'LEFT',42:'UP'}[t]
                            assert feasible(p,pair[0],OWN,a)
                            pair[0]=new_order(pair[0],OWN,a)
                        local,flows=after(local,pair);env.step(pair,[])
                        assert view(env)==fast(local)
                        if not intervention and enemy11=='recorded' and enemy10_now=='recorded':
                            assert view(env)==archived(f[t+1])
                        post=local.obs(0)['players']
                        tree=[(s,id,age) for s,z in enumerate(post) for id,(pt,age) in z[3].items() if pt==27]
                        assert len(tree)==1
                        trace.append(dict(t=t,cash=flows['cash'],margin=flows['margin'],payments=flows['flows'],
                                          killed=flows['events'],tree27=tree[0],official_tree_cash=local.obs(0)['trees'][tree[0][1]][1],orders=pair,absent_recorded_orders=absent,
                                          materially_new_workers=[sorted(set(pp[2])-set(rr[2])) for pp,rr in zip(post,f[t+1][0]['observation']['players'])],
                                          tracked={u:post[s][2].get(u) for s,u in ((0,OWN),(1,FOE),(1,GUARD))}))
                    roots.append(dict(intervention=intervention,enemy11_t41=enemy11,enemy10_t41=enemy10_now,enemy10_t42_if_survives=enemy10,trace=trace))
            rule.official_state.restore(env,frozen)
        sim.cash.events.clear();sim.step(*true);env.step(true,[])
        states.append(dict(t=start,margin=round(sim.cash[0]-sim.cash[1],2)))
    assert view(env)==fast(sim)==archived(f[-1]) and list(game['rewards'])==[s.reward for s in env.states]
    act={uid:act_history(f,41,s,uid,[-10,-5,-3,-2,-1,0]) for s,uid in ((0,OWN),(1,FOE),(1,GUARD))}
    return dict(replay_sha=hashlib.sha256(raw).hexdigest(),official_reproduction=299,source=str(SOURCE.relative_to(HERE.parent)),
                actual_public_root_players=f[41][0]['observation']['players'],action_history=act,branches=roots)
if __name__=='__main__':
    result=run();(HERE/'branches.json.gz').write_bytes(gzip.compress((json.dumps(result,separators=(',',':'))+'\n').encode(),mtime=0))
    for a,b in zip(result['branches'][::2],result['branches'][1::2]):
        print(a['enemy11_t41'],a['enemy10_t42_if_survives'],[(t['t'],t['tree27'][0],w['tree27'][0],round(w['margin']-t['margin'],2)) for t,w in zip(a['trace'],b['trace'])])
