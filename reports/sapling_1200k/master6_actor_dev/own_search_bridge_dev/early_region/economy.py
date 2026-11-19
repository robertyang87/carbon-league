"""Reconcile the incremental score with both sides' cash flows after the changed order."""
from collections import Counter
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'tree_response_training'))
import select_roots as sr
from turning_point_ledger import CashEvents

def ledger(entry,arm):
    out=HERE/f"seat{entry['seat']}"
    q=sr.load(out/'live_protocol.json')
    raw=sr.load(sr.ROOT/q['record']['source'])
    rec=sr.load(out/f'live_{arm}.json')
    sim=sr.FastCarbon(raw['game_configuration']).reset(seed=entry['seed'])
    for pair in raw['actions'][:entry['step']]:sim.step(*pair)
    start=list(sim.cash);sim.cash=CashEvents(start)
    money=[Counter(),Counter()];snapshots={}
    tail=sr.load(out/rec['action_tape'])
    for offset,pair in enumerate(tail,1):
        sim.cash.events.clear();sim.step(*pair)
        for kind,seat,amount,*rest in sim.cash.events:money[seat][kind]+=amount
        if offset in (1,10,30,len(tail)):
            for seat in (0,1):
                assert abs(sum(money[seat].values())+start[seat]-sim.cash[seat])<1e-5
            snapshots[str(offset)]=[dict(m) for m in money]
    assert list(sim.cash)==rec['score']
    return snapshots

def main():
    result=[]
    for entry in sr.load(HERE/'selection.json'):
        p,c=(ledger(entry,arm) for arm in ('parent','third'))
        s=entry['seat'];changes={}
        for offset in p:
            kinds=set().union(*p[offset],*c[offset])
            changes[offset]={k:sum((1 if j==s else -1)*(c[offset][j].get(k,0)-p[offset][j].get(k,0))
                                  for j in (0,1)) for k in sorted(kinds)}
        result.append(dict(seed=entry['seed'],seat=s,parent=p,third=c,change_in_margin_by_channel=changes))
    sr.retain(HERE/'cash.json',result)
if __name__=='__main__':main()
