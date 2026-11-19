"""Attribute each of the 113 recorded last-five-turn matched alternatives to its cash sources."""
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import sys
from economy_extract import CashEvents,FastCarbon,KEYS,POS,atomic,compact

HERE=Path(__file__).resolve().parent
def evaluate(raw):
    src=json.load(gzip.open(HERE/'plant_timing_final_branches.json.gz'))
    games=defaultdict(list);results=[]
    for r in src:games[r['id']].append(r)
    for gid,entries in games.items():
        env=json.load(gzip.open(raw/(gid+'.json.gz')))
        frames=env['steps'];cfg=env['configuration'];T=len(frames)-1
        sim=FastCarbon(cfg).load_observation(frames[0][0]['observation'])
        by_t=defaultdict(list)
        for r in entries:by_t[r['t']].append(r)
        for t in range(max(by_t)+1):
            for r in by_t[t]:
                s=r['seat'];uid=r['worker'];arms=[]
                for a,expect in zip(['NONE',r['preselected_move']],r['arms']):
                    branch=FastCarbon(cfg).set_state(sim.get_state());branch.cash=CashEvents(branch.cash);entries=[];flows=Counter()
                    for j in range(t,T):
                        jo=[dict(v['action'] or {}) for v in frames[j+1]]
                        if j==t:
                            if a=='NONE':jo[s].pop(uid,None)
                            else:jo[s][uid]=a
                        branch.cash.events.clear();branch.step(*jo)
                        for k,who,v,*detail in branch.cash.events:
                            flows[KEYS[POS[k]]]+=v if who==s else -v
                            if k in ('plant','seize') and who==s and detail[0]==uid:entries.append([j,k,round(-v,4),detail[1]])
                    if [round(v,2) for v in branch.cash]!=expect['cash']:raise ValueError(gid)
                    arms.append((flows,entries))
                d={key:round(arms[1][0][key]-arms[0][0][key],4) for key in KEYS}
                if abs(sum(d.values())-r['margin_delta'])>.005:raise ValueError('mismatch '+gid)
                results.append(dict(id=gid,t=t,seat=s,worker=uid,kind=r['purchase_kind'],branch_margin_delta=r['margin_delta'],
                                    difference_by_source=d,logged_purchases=arms[0][1],alternative_purchases=arms[1][1]))
            sim.step(*[v['action'] or {} for v in frames[t+1]])
    assert len(results)==len(src)
    report={}
    for key,rows in [('total',results)]+[(key,[r for r in results if r['kind']==key]) for key in ['plant','seize']]:
        n=len(rows);report[key]=dict(n=n,preselected_move_purchases_later=sum(bool(r['alternative_purchases']) for r in rows),
            preselected_move_higher=sum(r['branch_margin_delta']>0 for r in rows),move_avg_margin=round(sum(r['branch_margin_delta'] for r in rows)/n,3),
            move_avg_per_source={k:round(sum(r['difference_by_source'][k] for r in rows)/n,3) for k in KEYS})
    out=dict(scope='Conditional on recorded actions by every other worker and the observed future. Differences are preselected move minus recorded stay, oriented to the tested planter.',
        groups=report,cases=results)
    atomic(HERE/'plant_timing_final_flows.json.gz',gzip.compress(compact(out),mtime=0))
    return report
if __name__=='__main__':print(json.dumps(evaluate(Path(sys.argv[1])),ensure_ascii=False,indent=1))
