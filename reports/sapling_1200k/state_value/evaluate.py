"""One frozen evaluation; full row traces and map-clustered paired baseline errors."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from data import HERE, OLD, NEW, dataset, digest, splits
from value import CashMargin30, FEATURE_NAMES

NAMES=('head','zero','trend10','trend30')
def names(g): return g.get('models',g.get('seat_names'))
def rounded(x):return round(float(x),5)


def metrics(y,pred,mid,mask):
    ids=np.unique(mid[mask])
    # Reweight every map after seat/opponent/slice restriction.
    weights=np.zeros(len(y))
    for gid in ids:
        use=mask&(mid==gid)
        weights[use]=1/(len(ids)*np.count_nonzero(use))
    result={}
    for i,key in enumerate(NAMES):
        res=pred[:,i]-y
        nonzero=(y!=0)&mask
        result[key]={'rmse':rounded(np.sqrt(weights@(res**2))),'mae':rounded(weights@abs(res)),
                     'bias':rounded(weights@res),'sign_nonzero':rounded(weights[nonzero]@(np.sign(pred[nonzero,i])==np.sign(y[nonzero]))/weights[nonzero].sum()) if nonzero.any() else None}
    return {'maps':len(ids),'rows':int(mask.sum()),'estimates':result}


def score(split,items,folder,model):
    x,y,b,mid,seat,tick=dataset(items,folder)
    pred=np.column_stack(((x/model.scale)@model.coef,b))
    all_rows=np.ones(len(y),dtype=bool)
    report={'overall':metrics(y,pred,mid,all_rows),'by_seat':{},'by_opponent':{},'by_map':[],'by_cohort':{},'counterexamples':[]}
    for side in (0,1):report['by_seat'][side]=metrics(y,pred,mid,seat==side)
    rivals=np.asarray([names(items[j])[1-k] for j,k in zip(mid,seat)])
    for opp in sorted(set(rivals)):report['by_opponent'][opp]=metrics(y,pred,mid,rivals==opp)
    cohorts=np.asarray([items[j].get('cohort','archived Top5 mutual') for j in mid])
    for name in sorted(set(cohorts)):report['by_cohort'][name]=metrics(y,pred,mid,cohorts==name)
    rng=np.random.default_rng(53030)
    map_err=[];lines=['game\tmap_seed\tseat\tt\ttruth\thead\tzero\ttrend10\ttrend30']
    for j,item in enumerate(items):
        idx=np.where(mid==j)[0];use=mid==j
        er=((pred[idx]-y[idx,None])**2).mean(axis=0);map_err.append(er)
        per_seat=[metrics(y,pred,mid,use&(seat==side))['estimates']['head']['rmse'] for side in (0,1)]
        report['by_map'].append(dict(id=item['id'],seed=item['seed'],names=names(item),canonical_sha256=item['canonical_sha256'],
                                     **metrics(y,pred,mid,use),seat_head_rmse=per_seat))
        for i in idx:
            v=[y[i],*pred[i]]
            lines.append(f"{item['id']}\t{item['seed']}\t{seat[i]}\t{tick[i]}\t"+'\t'.join(f'{val:.6f}' for val in v))
    map_err=np.asarray(map_err);B=3000
    samples=rng.integers(0,len(items),(B,len(items)))
    # Paired map bootstrap. Report differences in RMSE after the equal-map MSE aggregation.
    draws=np.sqrt(map_err[samples].mean(axis=1))
    report['head_minus_baseline_cluster_ci']={}
    for j,name in enumerate(NAMES[1:],1):
        diff=draws[:,0]-draws[:,j];q=np.quantile(diff,[.025,.975])
        report['head_minus_baseline_cluster_ci'][name]={'rmse_diff':rounded(np.sqrt(map_err[:,0].mean())-np.sqrt(map_err[:,j].mean())),
                    'ci95':list(map(rounded,q)),'maps_with_lower_rmse':int((map_err[:,0]<map_err[:,j]).sum())}
    candidates=np.where((np.sign(pred[:,0])!=np.sign(y)) & (abs(y)>0) & (tick>=3))[0]
    ranking=sorted(candidates,key=lambda i:abs(pred[i,0]-y[i]),reverse=True)
    selected=set()
    for i in ranking:
        if mid[i] in selected:continue
        selected.add(mid[i]);j=int(mid[i]);s=int(seat[i]);v=x[i]/model.scale*model.coef
        ranking_fields=np.argsort(abs(v))[::-1][:6]
        report['counterexamples'].append({'id':items[j]['id'],'seat':s,'own':names(items[j])[s],
             'opponent':names(items[j])[1-s],'t':int(tick[i]),'forecast':rounded(pred[i,0]),'actual':rounded(y[i]),
             'trend10':rounded(pred[i,2]),'trend30':rounded(pred[i,3]),
             'largest_signed_linear_terms':{FEATURE_NAMES[k]:rounded(v[k]) for k in ranking_fields}})
        if len(selected)==3:break
    archive=HERE/(split+'_predictions.tsv.gz')
    archive.write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode(),mtime=0))
    report['row_archive_sha256']=digest(archive)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',required=True,type=Path);p.add_argument('--new',required=True,type=Path)
    a=p.parse_args()
    frozen=json.loads((HERE/'frozen.json').read_bytes())
    bound={'cash_margin_30.npz':'checkpoint_sha256','value.py':'runtime_sha256','data.py':'reader_sha256',
           'CONTRACT.md':'contract_sha256','train.py':'fit_sha256'}
    if any(digest(HERE/path)!=frozen[k] for path,k in bound.items()):
        raise ValueError('frozen evaluation input changed')
    sp=splits()
    if [v['id'] for v in frozen['train_games']]!=[g['id'] for g in sp['train']]:
        raise ValueError('training map identity changed')
    model=CashMargin30()
    report={'frozen_model':frozen['checkpoint_sha256'],'frozen_receipt':digest(HERE/'frozen.json'),'new_manifest_sha256':digest(NEW),
            'evaluator_sha256':digest(HERE/'evaluate.py')}
    for split,folder in [('old_val',a.old),('new_diag',a.new)]:
        report[split]=score(split,sp[split],folder,model)
        print(split,report[split]['overall'],flush=True)
    (HERE/'scores.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
