"""Compare the new site's whole income with the focal planter's actual next purchase after moving away."""
from fit import HERE,sr,CashEvents
from decision import _D
def one(out,arm):
    q=sr.load(out/'live_protocol.json');base=sr.load(sr.ROOT/q['record']['source']);cfg=base['game_configuration'];seat=q['seat'];root=q['root']
    sim=sr.FastCarbon(cfg).reset(seed=q['seed'])
    for pair in base['actions'][:root['t']]:sim.step(*pair)
    sim.cash=CashEvents(sim.cash)
    nearby=[v.id for v in sim.trees.values() if 0<_D[v.pos,root['cell']]<=2]
    info=sr.load(out/f'live_{arm}.json');tape=sr.load(out/info['action_tape'])
    tree=None;loss=None;worker_death=None;ground=tax=price=0.;buy=None;site=None;neighbors=[0.,0.]
    for offset,pair in enumerate(tape):
        t=sim.step_count;alive=root['worker'] in sim.workers
        sim.cash.events.clear();sim.step(*pair)
        if alive and root['worker'] not in sim.workers:worker_death=t
        for kind,side,value,*args in sim.cash.events:
            if tree is None and kind in ('plant','seize') and side==seat and args[0]==root['worker']:
                obj=next(v for v in sim.trees.values() if v.pos==args[1] and v.player==seat)
                tree,site,buy,price=obj.id,obj.pos,t,-value
            if kind=='tree_absorption':
                if args[0] in nearby and offset<50:neighbors[side]+=value-args[2]
                if args[0]==tree and loss is None and side==seat:ground+=value-args[2];tax+=args[2]
        if tree and loss is None and (tree not in sim.trees or sim.trees[tree].player!=seat):loss=sim.step_count
    return dict(arm=arm,t_first_acquire=buy,site=site,paid=price,ground=ground,tax=tax,end=loss,net=ground+tax-price,
                worker_death=worker_death,other_nearby_ground_next50=neighbors)
if __name__=='__main__':
    for s in (0,1):
        out=HERE/f'seat{s}'
        if all((out/f'live_{arm}.json').exists() for arm in ('parent','approach')):
            data=[one(out,a) for a in ('parent','approach')];sr.retain(out/'focal.json',data);print(s,data)
