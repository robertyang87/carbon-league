"""Reconstruct both teams' economics and verify each edited command against Sapling's actual decoder."""
from collections import Counter
from audit import HERE,sr
from bank import choose
from turning_point_ledger import CashEvents

def summary():
    rows=[]
    for result in sr.load(HERE/'scores.json'):
        seat=result['seat'];arm=result['arm'];seed=result['seed']
        directory=HERE/'games'/arm
        raw=sr.load(directory/f'raw_{seed}_{seat}.json.gz')
        record=sr.load(directory/f'row_{seed}_{seat}.json')
        wall=sr.load(directory/f'wall_{seed}_{seat}.json.gz')
        parent=sr.live.module(HERE.parent/'planter_capture_full_dev/candidate.py',f'bank_reference_{arm}_{seat}')
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=seed)
        sim.cash=CashEvents(sim.cash);events=[Counter(),Counter()];edits=[];windows=[]
        for t,pair in enumerate(raw['actions']):
            obs=sim.obs(seat);orders=parent.agent(obs,raw['game_configuration'])
            proposed=choose(obs,orders)
            if proposed and arm=='candidate':
                edits.append(proposed);orders=dict(orders,**{proposed['worker']:proposed['alternative']})
            assert orders==pair[seat],(arm,seat,t,orders,pair[seat])
            sim.cash.events.clear();sim.step(*pair)
            for kind,s,amount,*_ in sim.cash.events:events[s][kind]+=amount
            if (t+1)%50==0 or t+1==len(raw['actions']):
                windows.append(dict(step=t+1,cash=list(sim.cash),trees=[len(sim.obs(seat)['players'][j][3]) for j in (0,1)]))
        assert (any(raw['extinct_by_seat']) or list(sim.cash)==result['score_by_seat']) and not any(record['agent_errors_by_seat'])
        rows.append(dict(**result,edits=edits,events=[dict(x) for x in events],windows=windows,
                         search=wall['summary'],elapsed=record['wall_seconds']))
    sr.retain(HERE/'full_ledger.json',rows)
if __name__=='__main__':summary()
