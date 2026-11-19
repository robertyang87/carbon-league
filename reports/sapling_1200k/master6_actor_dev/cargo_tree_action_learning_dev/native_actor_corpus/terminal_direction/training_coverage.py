"""Check actual action activity on a fixed stratified subset of the full network's own training roots."""
from collections import defaultdict,Counter
from tempfile import TemporaryDirectory
from compare import HERE,CORPUS,ROOT,Path,read,write,sha,clean,packages
from fastsim import FastCarbon

def main():
    register=read(CORPUS/'training_roots.json.gz')
    cards={c['sha256']:c for c in read(CORPUS/'sources.json')['cards'] if c['split']=='train'}
    groups=defaultdict(list)
    for r in register:groups[r['game']].append(r)
    assert set(groups)==set(cards) and len(cards)==26
    out=[]
    with TemporaryDirectory(prefix='sapling-direction-training-') as directory:
        models,meta=packages(read(HERE/'protocol.json'),Path(directory))
        stored=read(HERE/'packages.json')
        assert all(meta['packages'][k]['sha256']==stored['packages'][k]['sha256'] for k in models)
        for game,roots in sorted(groups.items()):
            card=cards[game];raw=read(ROOT/card['path']);s=card['expert_seat']
            assert len(roots)==96 and sha(ROOT/card['path'])==game
            selected={r['t']:r for r in sorted(roots,key=lambda r:r['t'])[6::12]}
            assert len(selected)==8
            cfg={k:v for k,v in raw['game_configuration'].items() if k!='randomSeed'}
            sim=FastCarbon(raw['game_configuration']).reset(seed=raw['seed'],full_carbon=raw['initial_full_carbon'])
            for t in range(max(selected)+1):
                obs=sim.obs(s)
                if t==0 or t in selected:
                    orders={k:clean(m.agent(obs,cfg)) for k,m in models.items()}
                else:
                    for m in models.values():m._FEAT.observe(obs)
                if t in selected:
                    assert orders['center']==clean(selected[t]['native_parent_action'])
                    roles={uid:'base' for uid in obs['players'][s][1]}|{uid:w[2].lower() for uid,w in obs['players'][s][2].items()}
                    diff={k:dict(Counter(roles[u] for u in roles if a.get(u)!=orders['center'].get(u))) for k,a in orders.items()}
                    out.append(dict(seed=card['seed'],terrain=card['terrain'],seat=s,t=t,game=game,free_commands=orders,changed_roles=diff))
                if t<max(selected):sim.step(*raw['actions'][t])
    write(HERE/'training_coverage.json.gz',dict(source_registry=sha(CORPUS/'training_roots.json.gz'),package_registry=sha(HERE/'packages.json'),
         rule='From each chronological set of 96 actual training roots, take indices 6,18,30,42,54,66,78,90. No inspection of actions or labels in selection.',rows=out))
    print('training public roots checked',len(out))
if __name__=='__main__':main()
