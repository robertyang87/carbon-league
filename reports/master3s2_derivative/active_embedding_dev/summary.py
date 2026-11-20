from itertools import product
from statistics import mean,stdev
import hashlib,json
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
H=D/'hosted'
settle=(H/'settlement.json').read_bytes()
j=json.loads(settle);p=json.loads((D/'protocol.json').read_bytes())
assert j['records_reconciled'] and j['registered_games']==len(j['games'])==16
seeds=p['seeds']; arms=('parent','challenger'); m={}
for a in arms:
    games=[g for g in j['games'] if g['arm']==a]
    for seed in seeds:
        rows=sorted((g for g in games if g['seed']==seed),key=lambda g:g['seat'])
        assert [g['seat'] for g in rows]==[0,1]
        scores=[g['official_scores_by_seat'] for g in rows]
        margins=[v[i]-v[1-i] for i,v in enumerate(scores) if v is not None]
        m.setdefault(seed,{})[a]=dict(
            wins=sum(g['counted'] and g['winner_seat']==g['seat'] for g in rows),
            registered=2,unresolved=sum(not g['counted'] for g in rows),
            margin=mean(margins) if len(margins)==2 else None,
            win_fraction=sum(g['counted'] and g['winner_seat']==g['seat'] for g in rows)/2)
for seed in seeds:
    x=m[seed];x['challenger_minus_parent']={k:x['challenger'][k]-x['parent'][k] if x['challenger'][k] is not None and x['parent'][k] is not None else None for k in ('margin','win_fraction')}
    canon=next(r for r in j['results']['paired']['challenger_minus_parent']['by_physical_map'] if r['seed']==seed)
    for k,canonk in [('margin','margin_change'),('win_fraction','win_fraction_change')]:
        if x['challenger_minus_parent'][k] is None:assert canon[canonk] is None
        else:assert abs(x['challenger_minus_parent'][k]-canon[canonk])<1.e-8
stats={}
ids=tuple(range(4))
for a in (*arms,'challenger_minus_parent'):
    stats[a]={}
    for k in ('margin','win_fraction'):
        values=[m[s][a][k] for s in seeds]
        if any(x is None for x in values):
            stats[a][k]=None;continue
        centers=[mean(values[i] for i in sample) for sample in product(ids,repeat=4)]
        x=mean(values);e=stdev(values)/2*3.182446305284263
        zz={'mean':x,'bootstrap_percentiles_95':list(np.quantile(centers,(.025,.975))),'leave_one_map_out':{str(s):mean(v for i,v in enumerate(values) if j2!=i) for j2,s in enumerate(seeds)}}
        if k=='margin':zz['t_95_df3']=[x-e,x+e]
        stats[a][k]=zz
o=dict(schema='physical-map-replication-uncertainty/1',settlement_sha256=hashlib.sha256(settle).hexdigest(),map_order=seeds,physical_units=4,registered_games=16,statistics='Equal-weight paired seats per map; 256 ordered nonparametric cluster resamples with replacement, empirical quantiles .025/.975 (linear interpolation); continuous margin also shown with df=3 two-sided Student interval. Each leave-one-out mean retains both seats of the remaining three maps, six games per arm. Draws are zero wins. Small-sample diagnostic intervals only.',per_map=m,estimators=stats)
(H/'map_uncertainty.json').write_text(json.dumps(o,sort_keys=True,indent=2,allow_nan=False)+'\n')
