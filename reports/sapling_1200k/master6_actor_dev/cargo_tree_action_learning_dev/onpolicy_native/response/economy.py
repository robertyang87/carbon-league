"""Explain held outcomes with the two actually played response tapes, including contrary evidence."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from response import HERE,ROOT,base,read,sha
R=base.R

def arm(c,tag):
    src=read(ROOT/c['source']);s=c['seat']; t=c['root']['t']
    p=HERE/f"{c['id']}_{tag}.json.gz"
    r=read(p); sim=R.FastCarbon(src['game_configuration']).reset(seed=c['seed'])
    for pair in src['actions'][:t]:sim.step(*pair)
    visible=sim.obs(s)
    differing=set(c['root']['proposals']['parent'])|set(c['root']['proposals']['student'])
    departures={}
    for uid in sorted(differing):
        orig=c['root']['proposals']['parent'].get(uid,'STAY');new=c['root']['proposals']['student'].get(uid,'STAY')
        if orig!=new:departures[uid]=dict(before=orig,after=new,worker=visible['players'][s][2].get(uid),base=visible['players'][s][1].get(uid))
    sim.cash=R.CashEvents(sim.cash)
    rows=[]
    for i,pair in enumerate(r['tape']):
        o=sim.obs(s);side=o['players'][s];other=o['players'][1-s]
        before=list(sim.cash);sim.cash.events.clear();sim.step(*pair)
        events=[Counter(),Counter()]
        for kind,seat,delta,*_ in sim.cash.events:events[seat][kind]+=delta
        rows.append(dict(t=t+i,before=before,after=list(sim.cash),orders=pair,credit=events,
              counts=[{'workers':len(own[2]),'trees':len(own[3]),'cargo':round(sum(v[1] for v in own[2].values()),3)} for own in o['players']]))
    if not np.allclose(sim.cash,r['cash']['post'],atol=.011,rtol=0):raise ValueError('economic trace does not close')
    return rows,departures

def main():
    out=[]
    for c in read(HERE/'roots.json')['cases']:
        a,info=arm(c,'parent');b,otherinfo=arm(c,'student')
        if info!=otherinfo:raise ValueError('root mismatch')
        s=c['seat'];op=1-s;length=min(len(a),len(b))
        def diff(k):
            src=[a[0]['before'],b[0]['before']] if k==0 else [a[k-1]['after'],b[k-1]['after']]
            return (src[1][s]-src[1][op])-(src[0][s]-src[0][op])
        intervals=range(10,length+1)
        h=max(intervals,key=lambda k:abs(diff(k)-diff(k-10)))
        sums={}
        for tag,rows in [('parent',a),('student',b)]:
            sums[tag]=[Counter(),Counter()]
            for t in rows:
                for i in (0,1):sums[tag][i].update(t['credit'][i])
        ledger={}
        for event in set().union(*(x.keys() for both in sums.values() for x in both)):
            delta=[sums['student'][i][event]-sums['parent'][i][event] for i in (0,1)]
            ledger[event]=dict(own=delta[s],opponent=delta[op],relative=delta[s]-delta[op])
        out.append(dict(id=c['id'],start=c['root']['t'],seat=s,changed_orders=info,
                root_identical_enemy_order=a[0]['orders'][op]==b[0]['orders'][op],
                independent_enemy_order_differences=sum(x['orders'][op]!=y['orders'][op] for x,y in zip(a,b)),
                cash_credits_student_minus_parent=ledger,margin_change_by_end=diff(length),
                largest_absolute_ten_step_cash_gap_change=dict(ending_t=a[h-1]['t']+1,
                     difference_last_n={n:diff(h)-diff(h-n) for n in (3,5,10)},
                     parent=a[h-10:h],student=b[h-10:h])))
    R.write(HERE/'economy.json.gz',dict(roots=sha(HERE/'roots.json'), signed=sha(HERE/'result.json'),rows=out))
if __name__=='__main__':main()
