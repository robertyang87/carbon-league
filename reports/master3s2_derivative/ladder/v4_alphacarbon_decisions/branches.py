"""Bounded official interventions on an exposed ladder game; no new matches."""
from scan import *

FOE='player-1-worker-6'
OUR='player-0-worker-53'
def view(env):
    x=env.states[0].observation
    return [x.step,x.players,x.trees,x.full_carbon,[s.observation.carbon for s in env.states]]
def archived(f):
    x=f[0]['observation']
    return [x['step'],x['players'],x['trees'],x['full_carbon'],[s['observation']['carbon'] for s in f]]
def fast(s):
    x=s.obs(0)
    return [s.step_count,x['players'],x['trees'],list(s.carbon),[s.obs(k)['carbon'] for k in (0,1)]]
def signed_margin(x):
    return round(x[1][0][0]-x[1][1][0],2)
def hurry(p,who,team):
    u=p[0][2].get(who)
    if not u:return None
    homes=set(p[0][1].values())
    enemy_light=[w[0] for w in p[1][2].values() if w[2]=='COLLECTOR' and w[1]<=u[1]]
    # Move toward a home, preferring destinations untouched by a lighter publicly visible collector.
    options=[(int(any(dist(move(u[0],a),v)<=1 for v in enemy_light)),min(dist(move(u[0],a),b) for b in homes),i,a)
             for i,a in enumerate(('RIGHT','UP','LEFT','DOWN','NONE')) if feasible(p,team,who,a)]
    return min(options)[-1] if options else None
def collect():
    game=read();f=game['steps'];cfg=game['configuration'];terrain=f[0][0]['observation']['full_carbon']
    random.seed(cfg['randomSeed']);rule.np.random.seed(cfg['randomSeed'])
    env=rule.make('carbon',configuration=cfg);env.states[0].observation.full_carbon=copy.deepcopy(terrain)
    sim=rule.econ.FastCarbon(cfg).reset(full_carbon=terrain);sim.cash=rule.econ.CashEvents(sim.cash)
    cases=[]
    for t in range(299):
        true=[v['action'] or {} for v in f[t+1]]
        assert view(env)==fast(sim)==archived(f[t])
        if t in (267,274,279):
            env.steps[:-1]=[None]*t;env.logs=[];frozen=rule.official_state.take(env,[0,0])
            end=282 if t==267 else t+(3 if t==274 else 1)
            for change_at in ([None,'NONE','RIGHT','DOWN','LEFT'] if t==267 else [None]):
                for alt in (False,True):
                    rule.official_state.restore(env,frozen);curr=copy.deepcopy(sim);trace=[]
                    for step in range(t,end):
                        pair=[dict(v['action'] or {}) for v in f[step+1]];p=curr.obs(0)['players']
                        all_ids=[set(v[1])|set(v[2]) for v in p];omitted=[sorted(set(o)-pool) for o,pool in zip(pair,all_ids)]
                        pair=[{k:v for k,v in o.items() if k in pool} for o,pool in zip(pair,all_ids)]
                        if step==269 and change_at:
                            assert FOE in p[1][2]
                            pair[1]=new_order(pair[1],FOE,change_at)
                        if alt and t==267 and OUR in p[0][2]:
                            if step==267:a='DOWN'
                            elif step==268:a='LEFT'
                            elif step==269:a='NONE'
                            else:a=hurry(p,OUR,pair[0])
                            if a and feasible(p,pair[0],OUR,a):pair[0]=new_order(pair[0],OUR,a)
                        if alt and step==t and t!=267:
                            who='player-0-worker-40' if t==274 else 'player-0-worker-68'
                            a='RIGHT' if t==274 else 'DOWN'
                            assert feasible(p,pair[0],who,a);pair[0]=new_order(pair[0],who,a)
                        curr, flows=after(curr,pair)
                        env.step(pair,[])
                        assert view(env)==fast(curr)
                        if not alt and change_at is None:assert view(env)==archived(f[step+1])
                        cell=curr.obs(0)['players'][0][2].get(OUR)
                        src=f[step+1][0]['observation']['players']
                        new_ids=[sorted(set(q[2])-set(s[2])) for q,s in zip(curr.obs(0)['players'],src)]
                        trace.append(dict(step=step,actions=pair,absent_archival_entities=omitted,new_id_without_recorded_order=new_ids,
                                          cash=flows['cash'],margin=flows['margin'],bank=flows['bank'],flows=flows['flows'],
                                          killed=flows['events'],our53_after=cell,
                                          material_hash=hashlib.sha256(json.dumps(view(env),sort_keys=True,separators=(',',':')).encode()).hexdigest()))
                    cases.append(dict(t=t,until=end,replacement=alt,enemy_at_269=change_at or 'recorded',trace=trace))
            rule.official_state.restore(env,frozen)
        env.step(true,[]);sim.cash.events.clear();sim.step(*true)
    assert view(env)==fast(sim)==archived(f[-1]) and list(game['rewards'])==[s.reward for s in env.states]
    roots={}
    for k,us,them in ((267,OUR,FOE),(274,'player-0-worker-40','player-1-worker-56'),(279,'player-0-worker-68','player-1-worker-56')):
        ob=f[k][0]['observation']
        roots[str(k)]=dict(step=k,our_bases=ob['players'][0][1],enemy_bases=ob['players'][1][1],
                           our_history=act_history(f,k,0,us,[-10,-5,-3,-2,-1,0]),enemy_history=act_history(f,k,1,them,[-10,-5,-3,-2,-1,0]))
    return dict(public_roots_and_later_recorded_current_commands=roots,verified_official_steps=299,entire_replay_sha=hashlib.sha256(SRC.read_bytes()).hexdigest(),branches=cases)

if __name__=='__main__':
    data=collect()
    (HERE/'branches.json.gz').write_bytes(gzip.compress((json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n').encode(),mtime=0))
    for base,alt in zip(data['branches'][::2],data['branches'][1::2]):
        print('START',base['t'],base['enemy_at_269'])
        for a,b in zip(base['trace'],alt['trace']):
            if a['step']<271 or a['step'] in (274,276,279,280,281) or b['bank'][0]!=a['bank'][0]:print(a['step'],round(b['margin']-a['margin'],2),b['bank'],b['our53_after'],b['killed'])
