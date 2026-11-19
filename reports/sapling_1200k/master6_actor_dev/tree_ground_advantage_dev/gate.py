from fit import HERE,ACTOR,sr,CashEvents
import sys
sys.path.insert(0,str(ACTOR/'two_turn_intercept_dev'))
from ledger import check

def audit(out,q,arm):
    log=check(out,q['seat'],arm);raw=sr.load(sr.ROOT/q['record']['source']);s=q['seat'];cfg=raw['game_configuration']
    sim=sr.FastCarbon(cfg).reset(seed=q['seed'])
    for pair in raw['actions'][:q['root_t']]:sim.step(*pair)
    assert all(z.pos!=q['root']['cell'] for z in sim.trees.values())
    sim.cash=CashEvents(sim.cash);tree=None;payment=None;ground=tax=0.;team=[0.,0.];until=None
    rec=sr.load(out/f'live_{arm}.json');tape=sr.load(out/rec['action_tape'])
    for i,pair in enumerate(tape):
        sim.cash.events.clear();sim.step(*pair)
        for kind,side,value,*a in sim.cash.events:
            if i==0 and kind=='plant' and side==s and a==[q['root']['worker'],q['root']['cell']]:
                tree=next(x.id for x in sim.trees.values() if x.player==s and x.pos==q['root']['cell']);payment=-value
            if kind=='tree_absorption':
                team[side]+=value-a[2]
                if a[0]==tree and until is None and side==s:ground+=value-a[2];tax+=a[2]
        if tree and until is None and (tree not in sim.trees or sim.trees[tree].player!=s):until=sim.step_count
    return dict(ledger=log,site_tree=tree,paid=payment,site_ground=ground,site_tax=tax,tenure_end=until,team_ground=team)
def run():
    rows=[]
    for s in (0,1):
        out=HERE/f'seat{s}';q=sr.load(out/'live_protocol.json')
        a,b=[audit(out,q,arm) for arm in ('parent','approach')]
        assert a['paid'] is None
        rows.append(dict(seed=q['seed'],seat=s,t=q['root_t'],parent=a,changed=b,delta=b['ledger']['margin']-a['ledger']['margin']))
    ok=all(r['changed']['paid'] is not None and r['changed']['paid']>0 and r['delta']>0 for r in rows)
    sr.retain(HERE/'gate.json',dict(protocol=sr.sha(HERE/'protocol.json'),eligible=ok,rows=rows))
    return ok
if __name__=='__main__':print(run())
