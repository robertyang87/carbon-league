"""Summarize conditional, observable predictions. All intervals resample complete games."""
import argparse
from pathlib import Path
import gzip
import json
import numpy as np
from dataset import HERE, FEATURES, SPLITS, REASONS, read

TARGETS=['held_turns','gross_cash']


def main(cache,predictions):
    a=np.load(cache); xs=a['x']; yy=a['y']; y=yy[:,:2]
    preds=np.load(predictions); p=preds['pred']; mean=preds['mean']; ridge=preds['ridge']; raw=preds['raw']
    rowmap=a['map']; seat=a['seat']; kinds=xs[:,0]; reasons=a['reason']; rawprice=yy[:,4]
    maps=read(HERE/'maps.json.gz')['maps']; train_names={name for m in maps if m['split']=='train' for name in m['names']}
    split=np.array([SPLITS.index(r['split']) for r in maps])[rowmap]
    reserved=np.array([r['held_rival'] for r in maps])[rowmap,seat]
    opp=np.array([r['names'][::-1] for r in maps],dtype=object)[rowmap,seat]
    focal=np.array([r['names'] for r in maps],dtype=object)[rowmap,seat]
    test=(split>=2); masks={'held_map_all':split==2,'held_map_familiar_opponent':(split==2)&np.isin(opp,sorted(train_names)),
                            'held_names_facing_reserved':(split==3)&reserved,'held_names_reverse_perspective':(split==3)&~reserved,
                            'held_names_both_seats':split==3,'calibration':split==1}
    thresholds=[np.quantile(p[split==0,q],[.2,.4,.6,.8]) for q in (0,1)]
    def base(w):
        if not np.any(w):
            return {'n':0}
        out=dict(n=int(w.sum()),maps=int(len(np.unique(rowmap[w]))),nonrecoup=int(np.sum(y[w,1]<rawprice[w])),
                 mean_actual_price=float(rawprice[w].mean()),rival_worker_component=float(yy[w,2].mean()),collision_cash_component=float(yy[w,3].mean()),
                 ended={v:int((reasons[w]==i).sum()) for i,v in enumerate(REASONS)})
        for q,k in enumerate(TARGETS):
            act=y[w,q]; pred=p[w,q]; err=pred-act
            out[k]=dict(actual=float(act.mean()),predicted=float(pred.mean()),mae=float(np.abs(err).mean()),bias=float(err.mean()),rmse=float(np.sqrt(np.mean(err**2))),
                        raw_mae=float(np.abs(raw[w,q]-act).mean()),mean_reference_mae=float(np.abs(mean[w,q]-act).mean()),
                        same_input_ridge_mae=float(np.abs(ridge[w,q]-act).mean()))
        return out
    def group(w):
        d=base(w)
        if d['n']==0:
            return d
        mid, inv=np.unique(rowmap[w],return_inverse=True)
        for q,k in enumerate(TARGETS):
            groups=np.digitize(p[w,q],thresholds[q])
            d[k]['prediction_fifths']=[dict(n=int((groups==v).sum()),prediction=float(p[w,q][groups==v].mean()),
                 realized=float(y[w,q][groups==v].mean())) if np.any(groups==v) else {'n':0} for v in range(5)]
            losses=[np.bincount(inv,weights=np.abs(z[w,q]-y[w,q]))/np.bincount(inv) for z in [p,mean,ridge]]
            d[k]['map_balanced_mae']=float(losses[0].mean())
            rng=np.random.default_rng(4307)
            boot=rng.integers(0,len(mid),(1500,len(mid)))
            for idx,ref in [(1,'mean'),(2,'ridge')]:
                delta=losses[0]-losses[idx]
                d[k]['map_mae_difference_vs_'+ref]={'estimate':float(delta.mean()),'ci95':np.quantile(delta[boot].mean(1),[.025,.975]).tolist()}
        return d
    def info(i):
        m=maps[rowmap[i]]
        return dict(id=m['id'],seed=m['seed'],player=m['names'][seat[i]],opponent=m['names'][1-seat[i]],seat=int(seat[i]),root=int(a['root'][i]),
                    cell=int(a['cell'][i]),pre={k:float(xs[i,j]) if j else ['plant','seize'][int(xs[i,j])] for j,k in enumerate(FEATURES)},
                    official_charged=float(rawprice[i]),held_turns=float(y[i,0]),pred_held=float(p[i,0]),gross=float(y[i,1]),pred_gross=float(p[i,1]),
                    reason=REASONS[reasons[i]],net=float(y[i,1]-rawprice[i]))
    breakdown={}
    for k,w in masks.items():
        if k=='calibration': continue
        cuts={'plant':kinds==0,'seize':kinds==1,'actual_nonrecoup':y[:,1]<rawprice,'early_transfer':(reasons==1)&(y[:,0]<=5),
              'last_30':xs[:,4]<=30,'recommended_pay':p[:,1]>xs[:,1],'recommended_defer':p[:,1]<=xs[:,1]}
        breakdown[k]={label:base(w & v) for label,v in cuts.items()}
    all_maps={m['id']:m for m in maps}
    map_receipts=[]
    for ix,m in enumerate(maps):
        if m['split']=='train': continue
        w=rowmap==ix
        for s in (0,1):
            map_receipts.append(dict(id=m['id'],split=m['split'],seat=s,player=m['names'][s],opponent=m['names'][1-s],
                                     facing_reserved=m['held_rival'][s], metrics=base(w & (seat==s))))
    (HERE/'map_scores.jsonl.gz').write_bytes(gzip.compress(('\n'.join(json.dumps(d,separators=(',',':'),allow_nan=False) for d in map_receipts)+'\n').encode(),mtime=0))
    w=masks['held_names_facing_reserved']; idx=np.flatnonzero(w)
    falsep=idx[(p[idx,1]>xs[idx,1])&(y[idx,1]<rawprice[idx])]
    falsed=idx[(p[idx,1]<=xs[idx,1])&(y[idx,1]>rawprice[idx])]
    examples=dict(highest_overprediction_underwater=info(falsep[np.argmax(p[falsep,1]-y[falsep,1])]),
                  highest_underprediction_profitable=info(falsed[np.argmax(y[falsed,1]-p[falsed,1])]),
                  most_overpredicted_tenure=info(idx[np.argmax(p[idx,0]-y[idx,0])]))
    for k in ('held_map_all','held_names_facing_reserved'):
        v=masks[k]; observed_profitable=y[v,1]>rawprice[v]; recommend=p[v,1]>xs[v,1]
        breakdown[k]['paid_observation_classification']={f'{pr}_{ac}':int(((recommend==bool(i))&(observed_profitable==bool(j))).sum())
            for i,pr in enumerate(['defer','pay']) for j,ac in enumerate(['not_profitable','profitable'])}
    out=dict(source_maps=len(maps),source_clean_games=len(all_maps),events=len(y),counts={s:dict(source_maps=sum(m['split']==s for m in maps),
             **base(split==i)) for i,s in enumerate(SPLITS)},opponent_names_ever_in_train=sorted(train_names),
             reserved_names=sorted(set(opp[w])), exposure_cutpoints=[t.tolist() for t in thresholds],
             held={k:group(v) for k,v in masks.items()},strata=breakdown,adversarial_examples=examples,
             unknown_by_registered_opponent={str(n):base(w & (opp==n)) for n in sorted(set(opp[w]))},
             unknown_by_focal_registered_player={str(n):base(w & (focal==n)) for n in sorted(set(focal[w]))})
    (HERE/'observed.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cache',type=Path);ap.add_argument('predictions',type=Path);opts=ap.parse_args()
    main(opts.cache,opts.predictions)
