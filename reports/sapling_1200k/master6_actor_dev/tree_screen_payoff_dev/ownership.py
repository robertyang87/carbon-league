"""Measure the share of the next 16 tree payments actually collected by the opponent after a two-step takeover."""
from collections import defaultdict
from fit import HERE,ACTOR,sr,CashEvents,HORIZON
from statistics import median
def run():
    observations=sr.load(ACTOR/'tree_screen_response_dev/planned_threat_observations.json.gz')
    by=defaultdict(set)
    for row in observations:
        # A particular tree can attract multiple planters at the same time. Count that observation once.
        if row['seized']:by[row['source']].add((row['t'],row['tree']))
    games=[]
    for src,selections in by.items():
        raw=sr.load(sr.ROOT/src);s=raw['candidate_seat'];cfg=raw['game_configuration']
        sim=sr.FastCarbon(cfg).reset(seed=raw['seed']);sim.cash=CashEvents(sim.cash)
        receipt=defaultdict(list);changes=defaultdict(list)
        for t,joint in enumerate(raw['actions']):
            old={k:u.player for k,u in sim.trees.items()}
            sim.cash.events.clear();sim.step(*joint)
            for kind,side,val,*a in sim.cash.events:
                if kind=='tree_absorption':receipt[a[0]].append((t,side,val))
            for k,tree in sim.trees.items():
                if k in old and old[k]!=tree.player:changes[k].append((t,tree.player))
        cases=[]
        for t,tree in sorted(selections):
            inc=[sum(v for at,who,v in receipt[tree] if t<=at<t+HORIZON and who==i) for i in (s,1-s)]
            ret=next((at for at,side in changes[tree] if t+1<at<t+HORIZON and side==s),None)
            cases.append(dict(t=t,tree=tree,own=inc[0],enemy=inc[1],recaptured_at=ret))
        games.append(dict(source=src,seed=raw['seed'],seat=s,cases=cases))
    sr.retain(HERE/'seized_tree_tenures.json.gz',games)
    for tag,pred in [('train',lambda s:s<=621758326),('seen_evaluation',lambda s:s>621758326)]:
        rs=[r for g in games if pred(g['seed']) for r in g['cases']]
        total=sum(r['enemy']+r['own'] for r in rs);enemy=sum(r['enemy'] for r in rs)
        print(tag,len(rs),sum(r['recaptured_at'] is not None for r in rs),round(enemy/total,4),
            round(median(r['enemy'] for r in rs),2))
if __name__=='__main__':run()
