"""Verify both arms for each physical board and summarize early cash and establishment."""
import gzip
import json
from pathlib import Path
from statistics import mean

from build import HERE, sha

plan_bytes = (HERE/'protocol.json').read_bytes()
plan = json.loads(plan_bytes)
rows=[]
for ticket in plan['tickets']:
    seed, seat = ticket['seed'], ticket['seat']
    path = HERE/f'{seed}_{seat}.json.gz'
    source=path.read_bytes()
    receipt=json.loads(gzip.decompress(source))
    assert receipt['protocol']==sha(plan_bytes) and receipt['ticket']==ticket
    games=receipt['games']
    assert len(games)==2 and games['parent']['rows'][0]['orders']==games['candidate']['rows'][0]['orders']
    first_difference=None
    for a,b in zip(games['parent']['rows'],games['candidate']['rows'],strict=True):
        assert a['step']==b['step']
        if a['orders']!=b['orders']:
            first_difference=dict(step=a['step'],same_public=a['public']==b['public'],
                                  parent=a['orders'],candidate=b['orders'], cash_before=[
                                      games[k]['rows'][a['step']-1]['cash'] for k in ('parent','candidate')])
            break
    row=dict(seed=seed,seat=seat,sha256=sha(source),first_joint_order_difference=first_difference,arms={})
    for tag,arm in games.items():
        assert len(arm['rows'])==plan['horizon']
        clocks=[q['clocks'][seat] for q in arm['rows']]
        acc=arm['ledger']
        margin=lambda r:r['cash'][seat]-r['cash'][1-seat]
        row['arms'][tag]=dict(first_tree=arm['first_tree'][seat], enemy_first_tree=arm['first_tree'][1-seat],
                              ts={step:margin(arm['rows'][step-1]) for step in (25,50,100)},
                              trees_planted=arm['trees_planted'][seat], max_wall=max(c['wall'] for c in clocks),
                              above_one=sum(c['wall']>1 for c in clocks),
                              tree_net=sum(acc[seat].get(k,0)-acc[1-seat].get(k,0) for k in
                                           ('ground_rent','cargo_tax','tree_collision_cash','plant','seize')),
                              return_margin=acc[seat].get('own_delivery',0)-acc[1-seat].get('own_delivery',0))
    rows.append(row)
out=dict(plan=sha(plan_bytes),rows=rows,maps={})
for seed in sorted({r['seed'] for r in rows}):
    pair=[r['arms'] for r in rows if r['seed']==seed]
    assert len(pair)==2
    out['maps'][seed]={f't{n}_change':mean(q['candidate']['ts'][n]-q['parent']['ts'][n] for q in pair) for n in (25,50,100)}
(HERE/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
