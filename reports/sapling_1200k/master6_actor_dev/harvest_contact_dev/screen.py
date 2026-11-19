import copy
from scan import HERE,sr
from evade import intercept
def screen():
    result=[];w=sr.load(HERE/'weights.json')['w']
    for c in sr.load(HERE/'coverage.json.gz'):
        if c['seed']<621758325:continue
        raw=sr.load(sr.ROOT/c['source']);s=c['seat'];sim=sr.FastCarbon(raw['game_configuration']).reset(seed=c['seed']);prev=None;hits=[]
        for t,pair in enumerate(raw['actions'][:201]):
            obs=sim.obs(s);r=intercept(obs,pair[s],prev,w) if t>=50 else None;prev=obs
            if r:
                other=copy.deepcopy(sim);cmds=list(pair);a=dict(cmds[s]);new=r['alternative'];uid=r['worker']
                if new=='NONE':a.pop(uid,None)
                else:a[uid]=new
                cmds[s]=a;other.step(*cmds)
            sim.step(*pair)
            if r:hits.append(dict(root=r,original=list(sim.last_collisions),alternative=list(other.last_collisions)))
        result.append(dict(seed=c['seed'],seat=s,source=c['source'],hits=hits))
    sr.retain(HERE/'screen.json',result)
    for c in result:
        print(c['seed'],c['seat'],len(c['hits']),[(v['root']['t'],round(v['root']['p'],2),int(any(e[0]==v['root']['target'] and e[4]==c['seat'] for e in v['alternative']))) for v in c['hits']])
if __name__=='__main__':screen()
