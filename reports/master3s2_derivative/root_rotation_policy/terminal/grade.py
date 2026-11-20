"""Read frozen, already exposed terminal outcomes; cluster summaries on physical map."""
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent

def avg_ci(rows,fn):
    by_map=defaultdict(list)
    for r in rows:
        by_map[r['terrain']].append(fn(r))
    x=np.asarray([np.mean(values) for _,values in sorted(by_map.items())])
    if len(x) == 1:
        return {'n_maps':1,'equal_map_mean':float(x.mean()),'map_boot_percentile_95':None}
    rng=np.random.default_rng(162801)
    estimates=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    return {'n_maps':len(x),'equal_map_mean':float(x.mean()),'map_boot_percentile_95':np.quantile(estimates,[.025,.975]).tolist()}

def grade(rows):
    n=len(rows)
    accuracy=[sum(r['choice'][p]==int(r['better_B']) for r in rows if r['better_B'] is not None) for p in range(2)]
    wins=[sum(r['margins'][r['choice'][p]]>0 for r in rows) for p in range(2)]
    diffs=[r for r in rows if r['choice'][0]!=r['choice'][1]]
    counts=np.sum([r['counts'] for r in rows],axis=0)
    scores=np.asarray([r['oriented_log_gain'] for r in rows])
    wdl=[r for r in rows if r['wdl_differ']]
    def indexed(r,s):
        i=int(r['better_B'])
        return r['counts'][s][i],r['counts'][s][1-i]
    return dict(n=n,n_maps=len({r['terrain'] for r in rows}),n_non_tied=sum(r['better_B'] is not None for r in rows),higher_margin_raw=accuracy,
      wdl_crossing_n=len(wdl),actual_winner_if_force_prior_choice=wins,
      n_raw_choice_changed=len(diffs),n_changed_toward_higher=sum(x['true_margin_change']>0 for x in diffs),
      n_joint_odds_toward_best=int((scores>0).sum()),n_away=int((scores<0).sum()),n_equal=int((scores==0).sum()),
      oriented_policy_log_odds=avg_ci(rows,lambda r:r['oriented_log_gain']),
      paired_margin_delta_given_binary_choice=avg_ci(rows,lambda r:r['true_margin_change']),
      prior_choice_accuracy_delta=avg_ci(rows,lambda r:int(r['choice'][1]==r['better_B'])-int(r['choice'][0]==r['better_B'])),
      paired_decoded_different=sum(r['n_decoded_changed'] for r in rows),decoded_n=n*128,decoded_A_B_other=counts.tolist(),
      decoded_followup_available_by_root=[sum(sum(r['counts'][k][:2])>0 for r in rows) for k in range(2)],
      decoded_better_worse_other=[(np.sum([indexed(r,k) for r in rows],axis=0).tolist()+[int(counts[k][2])]) for k in range(2)],
      wdl_crossing_roots=[{'id':[r[k] for k in ('seed','seat','step')],'choice':r['choice'],'margins':r['margins'],'counts_A_B_other':r['counts']} for r in wdl])

def summarize():
    p=HERE/'results.json'
    rows=json.loads(p.read_bytes())['records']
    groups={'all':grade(rows)}
    for group in sorted({r['group'] for r in rows}):
        groups[group]=grade([r for r in rows if r['group']==group])
    skills=defaultdict(list)
    for row in rows:
        for f in row['features']:
            target=1 if row['better_B'] else 0
            change=(f['log_B_over_A'][1]-f['log_B_over_A'][0])*(2*target-1)
            skills[f['role']].append({'id':[row[k] for k in ('seed','seat','step')],'unit':f['uid'],
                        'better_action':f['B'] if target else f['A'],'other':f['A'] if target else f['B'],'oriented_change':change})
    return dict(results_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),group=groups,unit_contributions=skills)

if __name__=='__main__':
    summ=summarize()
    (HERE/'summary.json').write_text(json.dumps(summ,indent=2,allow_nan=False)+'\n')
