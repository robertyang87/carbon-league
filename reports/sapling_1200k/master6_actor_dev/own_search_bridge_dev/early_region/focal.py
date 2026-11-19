"""Track the actual planter and every subsequent first purchase after the experimental move."""
from economy import HERE,sr,CashEvents

def trace(entry,arm):
    s=entry['seat'];folder=HERE/f'seat{s}'
    q=sr.load(folder/'live_protocol.json');uid=q['root']['worker']
    raw=sr.load(sr.ROOT/q['record']['source']);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=entry['seed'])
    for pair in raw['actions'][:entry['step']]:sim.step(*pair)
    sim.cash=CashEvents(sim.cash)
    rec=sr.load(folder/f'live_{arm}.json');tape=sr.load(folder/rec['action_tape'])
    rows=[];purchases=[]
    for pair in tape:
        if uid not in sim.workers:break
        start=sim.step_count;pos=sim.workers[uid].pos
        sim.cash.events.clear();sim.step(*pair)
        for e in sim.cash.events:
            if e[0] in ('plant','seize') and e[1]==s and e[3]==uid:
                purchases.append(dict(step=start,cell=e[4],kind=e[0],cost=-e[2]))
        rows.append(dict(step=start,start_cell=pos,command=pair[s].get(uid,'NONE'),
                         survived=uid in sim.workers))
    return dict(worker=uid,actions_alive=len(rows),first_actions=rows[:10],
                final_action=rows[-1],purchases=purchases)
if __name__=='__main__':
    sr.retain(HERE/'focal.json',[dict(**r,arms={arm:trace(r,arm) for arm in ('parent','third')})
              for r in sr.load(HERE/'selection.json')])
