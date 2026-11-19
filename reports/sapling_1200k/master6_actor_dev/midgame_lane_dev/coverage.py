"""Retrospective next-turn comparison for every qualifying root on five original development maps."""
import copy,sys
from route import HERE,propose,apply
sys.path.insert(0,str(HERE.parent/'tree_response_training'))
import select_roots as sr
from turning_point_ledger import CashEvents
def events(sim):
    return list(sim.last_collisions)
def audit(path,seat):
    raw=sr.load(path);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
    sim.cash=CashEvents(sim.cash);hits=[];bank=[]
    for t,pair in enumerate(raw['actions']):
        root=propose(sim.obs(seat),pair[seat]) if 100<=t<=200 else None
        if root:
            # Deep copy from plain balances so the simulator cash tracer sees no synthetic writes.
            cash=sim.cash;sim.cash=list(cash)
            try:alt=copy.deepcopy(sim)
            finally:sim.cash=cash
            cmds=list(pair);cmds[seat]=apply(pair[seat],root);alt.step(*cmds)
        sim.cash.events.clear();sim.step(*pair)
        bank.extend((t,e[1],e[2],e[3]) for e in sim.cash.events if e[0]=='own_delivery')
        if root:
            root.update(original_collisions=events(sim),alternative_collisions=events(alt),
                        parent_target_alive=root['target'] in sim.workers,alternative_target_alive=root['target'] in alt.workers)
            hits.append(root)
    for v in hits:
        v['target_banks_next10']=[(t,val) for t,s,val,uid in bank if uid==v['target'] and v['t']<=t<v['t']+10]
    return dict(seed=raw['seed'],seat=seat,source=str(path.relative_to(sr.ROOT)),roots=hits)
if __name__=='__main__':
    cases=[]
    for folder in ('planter_capture_full_dev/games/candidate','capture_reach_dev/games/parent','intercept_response_dev/games/parent','bank_under_pressure_dev/games/parent','deadline_freight_dev/games/parent'):
        for path in sorted((HERE.parent/folder).glob('raw*.gz')):
            cases.append(audit(path,int(path.name.split('_')[-1].split('.')[0])))
    sr.retain(HERE/'coverage.json.gz',cases)
    for c in cases:print(c['seed'],c['seat'],len(c['roots']),sum(bool(r['target_banks_next10']) for r in c['roots']))
