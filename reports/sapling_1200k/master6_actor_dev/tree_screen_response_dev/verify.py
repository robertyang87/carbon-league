"""Read back the original opponent's next commands, tree tenure and both terminal rewards."""
from screen_trial import HERE,sr
import sys
sys.path.insert(0,str(HERE.parent/"two_turn_intercept_dev"))
from ledger import check, CashEvents

def run(out):
    q=sr.load(out/"live_protocol.json")
    raw=sr.load(sr.ROOT/q['record']['source'])
    root=q['root']
    arms=[]
    for arm in ('parent','approach'):
        rec=check(out,q['seat'],arm)
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=q['seed'])
        for joint in raw['actions'][:root['t']]:sim.step(*joint)
        tape=sr.load(out/sr.load(out/f'live_{arm}.json')['action_tape'])
        changes=[];first=[];last=q['seat']
        tree_paid=[0.,0.]
        sim.cash=CashEvents(sim.cash)
        for offset,pair in enumerate(tape):
            sim.cash.events.clear()
            sim.step(*pair)
            for kind,side,val,*details in sim.cash.events:
                if kind=='tree_absorption' and details[0]==root['tree']:
                    tree_paid[side]+=val
            tree=sim.trees.get(root['tree'])
            owner=tree.player if tree else None
            if owner!=last:
                changes.append(dict(t=root['t']+offset,owner_after=owner));last=owner
            if offset<=4:
                def position(uid):return sim.workers[uid].pos if uid in sim.workers else None
                first.append(dict(t=root['t']+offset,orders=[pair[q['seat']].get(root['worker']),pair[1-q['seat']].get(root['target'])],
                                  positions=[position(root['worker']),position(root['target'])],owner_after=owner))
        arms.append(dict(arm=arm,first_five=first,ownership_changes=changes,tree_absorption=tree_paid,ledger=rec))
    sr.retain(out.parent/"verified.json",arms)
    for a in arms:print(a['arm'],a['first_five'][:3],a['ownership_changes'],a['ledger']['score'],a['ledger']['calls'])
if __name__=="__main__":
    paths=[HERE/"seat0",HERE/"first_trigger/seat0",HERE/"opposite_map/seat1"]
    for path in paths:
        if (path/"live_approach.json").exists() and (path/"live_parent.json").exists():run(path)
