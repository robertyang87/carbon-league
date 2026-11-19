"""Resample complete maps inside each designated panel, pairing all three policies on every draw."""
import hashlib
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent

def paired():
    path = HERE / 'fit.json'
    fit = json.loads(path.read_bytes())
    result = {'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'unit':'one map, paired complete aggregate on identical actionable slots; percentile bootstrap conditional on this historical panel',
              'seed':2547601,'replicates':20000,'panels':{}}
    for name,panel in fit['scores'].items():
        games=panel['by_map']
        rng=np.random.default_rng(result['seed'])
        draw=rng.integers(len(games),size=(result['replicates'],len(games)))
        comparisons={}
        for ref in ['control','parent']:
            comp={}
            for f in ['ce','agreement']:
                d=np.array([g[f]['candidate']-g[f][ref] for g in games])
                lo,hi=np.quantile(d[draw].mean(1),[.025,.975])
                comp[f]={'mean':float(d.mean()),'ci95':[float(lo),float(hi)],'less_equal_greater':[int((d<0).sum()),int((d==0).sum()),int((d>0).sum())]}
            comparisons['candidate_minus_'+ref]=comp
        result['panels'][name]=comparisons
    (HERE/'paired.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
    paired()
