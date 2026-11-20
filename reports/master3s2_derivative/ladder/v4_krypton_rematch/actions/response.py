"""Official counterfactuals with a one-unit interception and explicit spawn-lineage and response assumptions."""
from scan_orders import *
from branches import random,view,fast,archived,hurry
UID='player-0-worker-68'
ENEMY56='player-1-worker-56'
ENEMY68='player-1-worker-68'
WIN=LADDER/'v4_observations/replays/2cba12ae1302e3e6392cf0bc74476b12-1.json.gz'
def is_live(p,u):
    return any(u in w[1] or u in w[2] for w in p)
def ids(p):
    return {k for side in p for group in side[1:3] for k in group}
def remap(live, archive, tag, mode, actions):
    if mode=='numeric':
        mapping={u:u for u in ids(live)}
    else:
        mapping=tag
    return [{k:v for k,kold in mapping.items() if k in set(live[s][1])|set(live[s][2]) and (v:=actions[s].get(kold))}
            for s in range(2)]
def update_tag(tag,p_before,p_after,a_before,a_after,t):
    births=[]
    for s in range(2):
        old=set(p_before[s][2]);compare_old=set(a_before[s][2])
        can={tuple([v[0],v[2]]):k for k,v in a_after[s][2].items() if k not in compare_old}
        for k,v in p_after[s][2].items():
            if k not in old:
                match=can.get((v[0],v[2]))
                tag[k]=match
                births.append([s,k,v[0],v[2],match])
    return births
def initial(env,f,start):
    env.steps[:-1]=[None]*start;env.logs=[]
    return rule.official_state.take(env,[0,0])

def simulate(path,seat,match_score,roots):
    game=json.loads(gzip.decompress(path.read_bytes()));f=game['steps'];cfg=game['configuration']
    assert [game['rewards'][seat],game['rewards'][1-seat]]==match_score
    random.seed(cfg['randomSeed']);rule.np.random.seed(cfg['randomSeed'])
    env=rule.make('carbon',configuration=cfg)
    terrain=copy.deepcopy(f[0][0]['observation']['full_carbon'])
    env.states[0].observation.full_carbon=terrain
    sim=rule.econ.FastCarbon(cfg).reset(full_carbon=terrain);sim.cash=rule.econ.CashEvents(sim.cash)
    results=[]
    for start in range(299):
        true=[r['action'] or {} for r in f[start+1]]
        assert view(env)==fast(sim)==archived(f[start])
        if start in roots:
            if start==286:
                scenarios=[(mode,h,r) for mode in ('lineage','numeric') for h in (ACTIONS if mode=='lineage' else ['NONE']) for r in (ACTIONS if mode=='lineage' else ['LEFT'])]
            else:scenarios=[('lineage','NONE','LEFT')]
            freeze=initial(env,f,start)
            for mode,guard,reply in scenarios:
                for change in (False,True):
                    rule.official_state.restore(env,freeze);local=copy.deepcopy(sim);tag={k:k for k in ids(local.obs(0)['players'])};out=[]
                    for t in range(start,299):
                        actual=[r['action'] or {} for r in f[t+1]];p=local.obs(0)['players']
                        ap=f[t][0]['observation']['players']
                        pair=remap(p,ap,tag,mode,actual)
                        if start==286 and t==288 and is_live(p,ENEMY68):pair[1]=new_order(pair[1],ENEMY68,guard)
                        if start==286 and t==289 and is_live(p,ENEMY56):pair[1]=new_order(pair[1],ENEMY56,reply)
                        if change:
                            if start==286 and is_live(p,UID):
                                a={286:'RIGHT',287:'RIGHT',288:'DOWN',289:'NONE'}.get(t)
                                if t>=290:a=hurry(p,UID,pair[0])
                                if a and feasible(p,pair[0],UID,a):pair[0]=new_order(pair[0],UID,a)
                            if start==289 and t==289:
                                a='DOWN';u='player-0-worker-66'
                                assert feasible(p,pair[0],u,a);pair[0]=new_order(pair[0],u,a)
                            if start==297 and t==297:
                                # Win control is seat 1. Remaining bilateral roster and original joint orders checked.
                                u='player-1-worker-51';assert u in p[1][2]
                                assert move(p[1][2][u][0],'NONE') not in [move(v[0],pair[1].get(k,'NONE')) for k,v in p[1][2].items() if k!=u]
                                pair[1]=new_order(pair[1],u,'NONE')
                        local,q=after(local,pair);env.step(pair,[])
                        assert view(env)==fast(local)
                        if not change and guard=='NONE' and reply=='LEFT':assert view(env)==archived(f[t+1])
                        post=local.obs(0)['players'];births=update_tag(tag,p,post,ap,f[t+1][0]['observation']['players'],t)
                        out.append(dict(t=t,actions=pair,actual_changed_worker=(UID if start==286 else 'player-0-worker-66' if start==289 else 'player-1-worker-51'),
                            worker_after=post[seat][2].get(UID if start==286 else 'player-0-worker-66' if start==289 else 'player-1-worker-51'),
                            bank=q['bank'],cash=q['cash'],events=q['events'],flows=q['flows'],birth_mapping=births,
                            step_cash_margin=round(q['cash'][seat]-q['cash'][1-seat],2),
                            board_hash=hashlib.sha256(json.dumps(view(env),sort_keys=True,separators=(',',':')).encode()).hexdigest()))
                    assert env.done and all(o.status=='DONE' for o in env.states)
                    scores=[r.reward for r in env.states]
                    results.append(dict(start=start,mode=mode,guard288=guard,reply289=reply,changed=change,
                            official_scores_by_seat=scores,official_margin=round(scores[seat]-scores[1-seat],2),trace=out))
            rule.official_state.restore(env,freeze)
        sim.cash.events.clear();sim.step(*true);env.step(true,[])
    assert view(env)==fast(sim)==archived(f[-1]) and list(game['rewards'])==[s.reward for s in env.states]
    return dict(source_sha=hashlib.sha256(path.read_bytes()).hexdigest(),seat=seat,actual_scores_by_seat=game['rewards'],verified_steps=299,results=results)
if __name__=='__main__':
    scores={'rematch':simulate(SOURCE,0,[6245.53,6493.12],{286,289}),
            'earlier':simulate(WIN,1,[8871.56,8739.27],{297})}
    (HERE/'paths.json.gz').write_bytes(gzip.compress((json.dumps(scores,separators=(',',':'))+'\n').encode(),mtime=0))
    compact={}
    for name,s in scores.items():
        rows=[]
        for a,b in zip(s['results'][::2],s['results'][1::2]):
            row={k:a[k] for k in ('start','mode','guard288','reply289')}
            row.update(original=a['official_margin'],alternative=b['official_margin'],change=round(b['official_margin']-a['official_margin'],2))
            print(name,row);rows.append(row)
        compact[name]=rows
    (HERE/'terminal.json').write_text(json.dumps(compact,indent=2)+'\n')
