"""Count other actual purchases by the same worker, including their first-tenure ground returns."""
from study import HERE,V1,sr
from funnel import CashEvents

def one(q,name):
    s=q["seat"];r=q["root"];uid=r["worker"];t=r["t"]
    raw=sr.load(sr.ROOT/q['record']['source']);cfg=raw['game_configuration']
    p=HERE/f'seat{s}';doc=sr.load(p/f'live_{name}.json');f=p/doc['action_tape']
    if sr.sha(f)!=doc['action_sha256']:raise ValueError("stale continuation")
    sim=sr.FastCarbon(cfg).reset(seed=q["seed"])
    for pair in raw['actions'][:t]:sim.step(*pair)
    trs={};subsequent_seizures=0
    for i,pair in enumerate(sr.load(f)):
        sim.cash=CashEvents(sim.cash);sim.step(*pair)
        for typ,owner,v,*args in sim.cash.events:
            if owner!=s:continue
            if typ=="plant" and args[0]==uid:
                pos=args[1];birth=[tr for tr in sim.trees.values() if tr.player==s and tr.pos==pos and tr.age==1]
                if len(birth)!=1:raise ValueError("birth mismatch")
                tree=birth[0].id;trs[tree]=dict(t=t+i,uid=uid,tree=tree,cell=pos,price=-v,ground=0.,cargo_tax=0.,first_tenure_closed=False)
            if typ=="seize" and args[0]==uid:subsequent_seizures+=1
            if typ=="tree_absorption" and args[0] in trs and not trs[args[0]]['first_tenure_closed']:
                target=trs[args[0]];target['ground']+=v-args[2];target['cargo_tax']+=args[2]
        for tree,target in trs.items():
            if tree not in sim.trees or sim.trees[tree].player!=s:target['first_tenure_closed']=True
    births=list(trs.values())
    return dict(purchases=births,total_net_ground=sum(tr['ground']-tr['price'] for tr in births),subsequent_seizures=subsequent_seizures)
def run():
    rows=[]
    for q in sr.load(HERE/'selected_sites.json'):
        p=one(q,'parent');c=one(q,'approach')
        rows.append(dict(seed=q['seed'],seat=q['seat'],parent=p,candidate=c,delta_same_worker_ground_less_prices=c['total_net_ground']-p['total_net_ground']))
    sr.retain(HERE/'worker_substitutions.json',rows)
if __name__=='__main__':run()
