from measure import HERE,sr
from approach import choose
from guard_tree import _D
def run():
    rows=[]
    for g in sr.load(HERE/'paired.json.gz'):
        raw=sr.load(sr.ROOT/g['path']);s=g['seat'];cfg=raw['game_configuration']
        sim=sr.FastCarbon(cfg).reset(seed=raw['seed']);alt=sr.FastCarbon(cfg).reset(seed=raw['seed']);prev={}
        for t,cmd in enumerate(raw['actions']):
            obs=sim.obs(s);q=choose(obs,cmd[s],cfg,prev);prev=obs['players'][1-s][2]
            if q:
                alt.set_state(sim.get_state());ac=[dict(a) for a in cmd];ac[s][q['uid']]=q['after'];alt.step(*ac)
            sim.step(*cmd)
            if q:
                def status(world):
                    en=world.workers.get(q['enemy']);covered=any(w.player==s and w.is_collector and _D[w.pos,q['cell']]<=1 for w in world.workers.values())
                    return dict(covered=covered,planter_on=en is not None and en.pos==q['cell'])
                base=status(sim);change=status(alt)
                qualifies=not base['covered'] and change['covered'] and change['planter_on'] and base['planter_on']
                rows.append(dict(source=g['path'],source_sha256=g['source_sha256'],t=t,seat=s,seed=g['seed'],choice=q,base=base,changed=change,qualifies=qualifies))
    sr.retain(HERE/'approach.json.gz',rows)
    selected=next((r for r in rows if r['qualifies']),None)
    if selected:sr.retain(HERE/'selected.json',selected)
    print(len(rows),sum(r['qualifies'] for r in rows),selected)
if __name__=='__main__':run()
