"""Reconcile the v1 vs N34 replay, including public moves before large bank transfers."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
LADDER=HERE.parents[1]
sys.path.insert(0,str(LADDER / 'losses_nov20'))
from compare import summarize
import validation

card=json.loads((HERE/'scorecard.json').read_bytes())
game=card['id']
ticket=card['result'][0]
assert card['name_a']=='安生N34' and card['b']=='b99742a4a510a4de5dc99639b3c85604'
assert card['name_b']=='ds-pro-v1' and ticket['order']==['a','b'] and ticket['statuses']=={'a':'DONE','b':'DONE'}
name=game+'-1.json.gz'
raw=(LADDER/'replays'/name).read_bytes()
env=json.loads(gzip.decompress(raw))
assert env['rewards']==[ticket['scores'][k] for k in ticket['order']]
authority=validation.official({'path':name},LADDER/'replays')
assert authority=={'matched':299, 'first_divergence':None}, authority
res=summarize('安生N34',game,1)
frames=env['steps']
hist=res['steps'];banks=res['receipts_by_transition']
def before(tick,seat,unit):
    seq=[]
    for t in sorted({max(0,tick-k) for k in (10,5,3,1,0)}):
        ps=frames[t][0]['observation']['players']
        w=ps[seat][2].get(unit)
        seq.append(dict(t=t,worker=w,order=(frames[t+1][seat]['action'] or {}).get(unit,'NONE')))
    return seq
windows=[]
for start in range(290):
    end=start+10
    by_seat=[sum(e['amount'] for row in banks[start:end] for e in row if e['seat']==s) for s in (0,1)]
    windows.append(dict(start=start,end=end,margin_swing=round(hist[end]['margin']-hist[start]['margin'],2),
                        deposited_by_seat=[round(v,2) for v in by_seat]))
worst=min(windows,key=lambda v:v['margin_swing']);best=max(windows,key=lambda v:v['margin_swing'])
def annotate(win):
    win['deliveries']=[]
    for t in range(win['start'],win['end']):
        for e in banks[t]:
            win['deliveries'].append(dict(**e,t=t+1,history=before(t,e['seat'],e['worker'])))
for row in (worst,best):annotate(row)
res.update(official_board=authority,worst_10=worst,best_10=best)
out=json.dumps(res,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
(HERE/'ledger.json.gz').write_bytes(gzip.compress(out,mtime=0))
if __name__=='__main__':
    print(res['sign_counts'],res['longest_deficit'],res['worst'])
    print([(r['end'],r['end_margin'],r['difference']['bank'],r['difference']['net_tree']) for r in res['phases']])
    print(res['whole']['income'])
    print([(r['planted'],r['repaid']) for r in res['planting']])
    for w in (worst,best): print(w['start'],w['end'],w['margin_swing'],w['deposited_by_seat'],[(e['t'],e['worker'],e['amount']) for e in w['deliveries']])
