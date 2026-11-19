"""Test whether a lighter worker could reach the actual landing before two decisive opponent deposits."""
from measure import HERE,sr,_TO,_DIRECTION
def run():
    out=[]
    for r in sr.load(HERE/'turning_points_candidate.json.gz'):
        s=r['seat'];ranked=sorted((v for v in r['deliveries'] if v['seat']!=s),key=lambda x:-x['amount'])
        target=ranked[0];path=HERE/f"games/candidate/raw_{r['seed']}_{s}.json.gz";raw=sr.load(path)
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=r['seed']);alt=sr.FastCarbon(raw['game_configuration']).reset(seed=r['seed'])
        for t,pair in enumerate(raw['actions'][:target['t']+1]):
            obs=sim.obs(s);foe=obs['players'][1-s];own=obs['players'][s]
            uid=target['worker'];v=foe[2].get(uid)
            if t>=r['start']-10 and v:
                land=int(_TO[v[0],_DIRECTION.index(pair[1-s].get(uid))])
                options=[]
                for k,w in own[2].items():
                    if w[2]!='COLLECTOR' or w[1]>=v[1] or land not in _TO[w[0]]:continue
                    cmd=_DIRECTION[list(_TO[w[0]]).index(land)];ca=[dict(z) for z in pair]
                    if cmd:ca[s][k]=cmd
                    else:ca[s].pop(k,None)
                    alt.set_state(sim.get_state());alt.step(*ca)
                    options.append(dict(worker=k,act=cmd,original=pair[s].get(k),
                        changed=ca[s]!=pair[s],own_alive=k in alt.workers,foe_alive=uid in alt.workers,
                        own_cargo=alt.workers[k].carbon if k in alt.workers else None,cash=list(alt.cash)))
                out.append(dict(seed=r['seed'],seat=s,t=t,foe=uid,cargo=v[1],before=v[0],actual_landing=land,
                         on_enemy_base=land in foe[1].values(),direct_options=options))
            sim.step(*pair)
    sr.retain(HERE/'actual_return_reach.json',out)
    print([(r['seat'],r['t'],r['on_enemy_base'],r['direct_options']) for r in out if r['direct_options']])
if __name__=='__main__':run()
