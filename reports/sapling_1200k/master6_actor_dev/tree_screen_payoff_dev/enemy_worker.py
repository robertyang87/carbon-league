"""Follow the particular threatening planter in each genuinely replanned branch and price its subsequent actions."""
from fit import HERE,sr,CashEvents
def examine(out,arm):
    q=sr.load(out/'live_protocol.json');r=q['root'];s=q['seat'];other=1-s;raw=sr.load(sr.ROOT/q['record']['source'])
    sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
    for pair in raw['actions'][:r['t']]:sim.step(*pair)
    tape=sr.load(out/sr.load(out/f'live_{arm}.json')['action_tape'])
    sim.cash=CashEvents(sim.cash);death=None;actions=[];trees={};receipts=[];changes=[];last=s
    for j,pair in enumerate(tape):
        t=r['t']+j;alive=r['target'] in sim.workers;sim.cash.events.clear();sim.step(*pair)
        if alive and r['target'] not in sim.workers:death=t
        tree=sim.trees.get(r['tree']);owner=tree.player if tree else None
        if owner!=last:changes.append(dict(t=t,holder=owner));last=owner
        for kind,side,v,*a in sim.cash.events:
            if kind in ('plant','seize') and side==other and a[0]==r['target']:
                hit=next(x for x in sim.trees.values() if x.pos==a[1])
                trees[hit.id]=t;actions.append(dict(t=t,operation=kind,value=v,tree=hit.id))
            if kind=='tree_absorption' and a[0] in trees and side==other:
                receipts.append(dict(t=t,value=v,tree=a[0]))
        for tree_id in list(trees):
            obj=sim.trees.get(tree_id)
            if obj is None or obj.player!=other:del trees[tree_id]
    totals=[]
    for horizon in (16,50,299):
        start=r['t'];entry=sum(a['value'] for a in actions if a['t']<start+horizon)
        income=sum(a['value'] for a in receipts if a['t']<start+horizon)
        totals.append(dict(horizon=horizon,invested=-entry,first_tenure_income=income,direct_net=entry+income))
    return dict(arm=arm,seed=q['seed'],seat=s,t=r['t'],enemy_planter=r['target'],enemy_death=death,
                owner_changes=changes,actions=actions,first_tenure_payments=receipts,summaries=totals)
if __name__=='__main__':
    for seat in (1,0):
        out=HERE/f'seat{seat}'
        if (out/'live_approach.json').exists() and (out/'live_parent.json').exists():
            records=[examine(out,arm) for arm in ('parent','approach')];sr.retain(out/'enemy_worker.json',records)
            print(seat,[(r['arm'],r['enemy_death'],[round(x['direct_net'],2) for x in r['summaries']]) for r in records])
