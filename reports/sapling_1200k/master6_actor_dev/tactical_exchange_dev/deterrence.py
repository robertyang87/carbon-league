"""Tabulate real planter decisions conditional on distance to the closest defending collector."""
from collections import Counter
from measure import HERE,sr
from guard_tree import _D


def run():
    games=[]
    for g in sr.load(HERE/'paired.json.gz'):
        raw=sr.load(sr.ROOT/g['path']);cfg=raw['game_configuration'];s=g['seat']
        sim=sr.FastCarbon(cfg).reset(seed=raw['seed']);rows=[]
        for t,pair in enumerate(raw['actions']):
            obs=sim.obs(s);own=obs['players'][s];enemy=obs['players'][1-s];opps=[]
            trees={v[0]:(k,v[1]) for k,v in own[3].items()}
            for uid,(p,_,kind) in enemy[2].items():
                if kind!='PLANTER' or p not in trees or enemy[0]<cfg['seizeCost']:continue
                tree,age=trees[p]
                if age>=cfg['treeLifespan']:continue
                nearest=min((int(_D[p,v[0]]) for v in own[2].values() if v[2]=='COLLECTOR'),default=100)
                opps.append(dict(t=t,uid=uid,tree=tree,d=nearest,stay=not pair[1-s].get(uid)))
            sim.step(*pair)
            for r in opps:
                tree=sim.trees.get(r['tree']);r.update(seized=tree is not None and tree.player==1-s);rows.append(r)
        games.append(dict(source=g['path'],rows=rows))
    sr.retain(HERE/'real_opportunities.json.gz',games)
    counts=Counter()
    for g in games:
        for r in g['rows']:
            group='can_reach' if r['d']<=1 else 'cannot_reach'
            counts[group,'rows']+=1;counts[group,'stay']+=r['stay'];counts[group,'seized']+=r['seized']
    print(counts)
if __name__=='__main__':run()
