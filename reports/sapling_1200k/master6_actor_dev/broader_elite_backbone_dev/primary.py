"""Apply the prespecified actionable-slot paired map mean and disclose role and phase incidence."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from freeze_data import HERE,sha,save,load

def totals(z):
    a=Counter()
    for v in z['roles'].values(): a.update(v)
    return a

def describe(result):
    rows=[totals(x) for x in result['maps']];rows=[r for r in rows if r['free_both']]
    edges=np.asarray([[(r['parent_agreement']/r['free_both']),(r['candidate_agreement']/r['free_both'])] for r in rows])
    delta=edges[:,1]-edges[:,0];rng=np.random.default_rng(7924059)
    ci=np.quantile(rng.choice(delta,(10000,len(delta)),replace=True).mean(1),[.025,.975]).tolist()
    return dict(maps=len(rows),macro_parent=float(edges[:,0].mean()),macro_candidate=float(edges[:,1].mean()),
                macro_gain=float(delta.mean()),descriptive_paired_map_bootstrap_95=ci,map_positive=int((delta>0).sum()),
                map_negative=int((delta<0).sum()),map_equal=int((delta==0).sum()),micro=dict(sum(rows,Counter())),
                observed_joint=result['overall']['total'],by_role=result['macro_terrain_equal'])
def main():
    results={}
    for label,path in [('deployed',HERE/'serving.json'),('standalone',HERE/'standalone_serving.json')]:
        scope=load(path);results[label]={'sha256':sha(path.read_bytes()),'splits':{k:describe(v) for k,v in scope['results'].items()}}
    save(HERE/'primary.json',dict(aggregation_sha256=sha((HERE/'primary_metric.json').read_bytes()),models=results,
                                  scope='Actual expert action agreement on sampled replay positions; game utility and survival require active-play evidence.'))
    for a,r in results.items():
        for s,v in r['splits'].items():print(a,s,v['maps'],v['macro_parent'],v['macro_candidate'],v['macro_gain'],v['descriptive_paired_map_bootstrap_95'],v['map_positive'],v['map_negative'])
if __name__=='__main__':main()
