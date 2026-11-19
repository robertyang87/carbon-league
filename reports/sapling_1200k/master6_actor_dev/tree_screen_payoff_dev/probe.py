from fit import HERE,ACTOR,sr,price,HORIZON,CashEvents
from approach import choose

def signed_branch(out,arm,proto,raw):
    t=proto['root_t'];root=proto['root'];s=proto['seat'];sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
    for j in raw['actions'][:t]:sim.step(*j)
    cell=sim.trees[root['tree']].pos;sim.cash=CashEvents(sim.cash)
    tape=sr.load(out/sr.load(out/f'live_{arm}.json')['action_tape']);cash=dict(tree=0.,fee=0.,courier=0.)
    for pair in tape[:HORIZON]:
        active_tree = sim.trees.get(root['tree'])
        can_be_seized = active_tree is not None and active_tree.age < sim.config['treeLifespan']
        sim.cash.events.clear();sim.step(*pair)
        for kind,side,v,*args in sim.cash.events:
            sign=1 if s==side else -1
            if kind=='tree_absorption' and args[0]==root['tree']:cash['tree']+=sign*v
            if kind=='seize' and can_be_seized and args[1]==cell:cash['fee']+=sign*v
            if kind=='own_delivery' and args[0]==root['worker']:cash['courier']+=sign*v
    return cash

def run():
    weights=sr.load(HERE/'weights.json');results=[];history={}
    search=ACTOR/'tree_screen_response_dev'
    for out in (search/'seat0',search/'first_trigger/seat0',search/'opposite_map/seat1'):
        p=sr.load(out/'live_protocol.json');raw=sr.load(sr.ROOT/p['record']['source'])
        a,b=[signed_branch(out,arm,p,raw) for arm in ('parent','approach')]
        history[p['record']['source'],p['root_t']]=dict(parent=a,screen=b,increment=sum(b.values())-sum(a.values()))
    for g in sr.load(ACTOR/'tactical_exchange_dev/paired.json.gz'):
        raw=sr.load(sr.ROOT/g['path']);cfg=raw['game_configuration'];s=g['seat']
        sim=sr.FastCarbon(cfg).reset(seed=g['seed']);prev={}
        for t,pair in enumerate(raw['actions']):
            ob=sim.obs(s);r=choose(ob,pair[s],cfg,prev)
            if r and t+2<=299:
                ans=price(ob,pair[s],cfg,r,prev,weights)
                results.append(dict(t=t,seed=g['seed'],seat=s,source=g['path'],pack=raw['pack_sha256'],root=r,
                            score=ans,branch=history.get((g['path'],t))))
            prev=ob['players'][1-s][2];sim.step(*pair)
    sr.retain(HERE/'probe.json',results)
    print('accepted:',sum(r['score']['accept'] for r in results),'/',len(results))
    for r in results:print(r['seed'],r['seat'],r['t'],r['score']['accept'],round(r['score']['net'],2),str(r['branch'])[:260] if r['branch'] else '')
if __name__=='__main__':run()
