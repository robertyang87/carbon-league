"""Audit direct tree receipts and routing decisions separately on the eight held maps."""
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
METHODS=('original','bc','weighted')
DIRECTIONS=('NONE','UP','RIGHT','DOWN','LEFT')

def views(pairs):
    if not pairs:return {'n':0}
    changes=[]; improves=harms=0
    for c,r in pairs:
        d=r['logp']['weighted']-r['logp']['bc']
        changes.append(dict(match=c['match'],opponent=c['opponent'],t=r['t'],uid=r['uid'],cell=r['cell'],
                     issued=DIRECTIONS[r['y']],kind=r['kind'],lag=r['lag'],net=r['net'],delta=d,
                     proposal_p={k:float(np.exp(v)) for k,v in r['logp'].items()},
                     top1={k:DIRECTIONS[v] for k,v in r['top1'].items()}))
        improves+=int(r['top1']['weighted']==r['y'] and r['top1']['bc']!=r['y'])
        harms+=int(r['top1']['weighted']!=r['y'] and r['top1']['bc']==r['y'])
    values=np.array([v['delta'] for v in changes])
    return dict(n=len(changes),increased=int(np.sum(values>1e-10)),decreased=int(np.sum(values< -1e-10)),
                stable=int(np.sum(abs(values)<=1e-10)),weighted_top1_gains=improves,weighted_top1_losses=harms,
                delta_logp_mean=float(values.mean()),delta_logp_median=float(np.median(values)),
                biggest_increase=max(changes,key=lambda v:v['delta']),biggest_decrease=min(changes,key=lambda v:v['delta']))

def main():
    cards=json.loads(gzip.decompress((HERE/'held_rows.json.gz').read_bytes()))
    allpairs=[(g,r) for g in cards for r in g['rows']]
    buckets={}
    for kind in ('plant','seize','all'):
        for lag in (0,1,2):
            for label in ('profit','loss'):
                values=[(g,r) for g,r in allpairs if r['lag']==lag and (kind=='all' or r['kind']==kind) and
                         (r['net']>0 if label=='profit' else r['net']<0)]
                buckets[f'{kind}_{lag}_{label}']=views(values)
    buckets['ordinary']=views([(g,r) for g,r in allpairs if r['lag'] is None])
    held=json.loads((HERE/'held.json').read_bytes())['by_game']
    gap=np.array([g['nll']['weighted']-g['nll']['bc'] for g in held])
    # Only an uncertainty description; no model selection or change to the signed acceptance gate.
    ix=np.random.default_rng(551187).integers(len(gap),size=(10000,len(gap)))
    result=dict(slices=buckets,map_cluster_bootstrap_nll_gap=dict(weighted_minus_bc=float(gap.mean()),
                maps_improved=int(np.sum(gap<0)),n_maps=len(gap),percentile95=np.quantile(gap[ix].mean(1),[.025,.975]).tolist()))
    (HERE/'diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
