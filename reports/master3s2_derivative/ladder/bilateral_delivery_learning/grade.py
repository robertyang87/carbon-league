"""Map-equal, supervised held results and the independent natural response comparisons."""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np
from model import predict
from register import H, read, dump, sha

def report_held(rs):
    groups=defaultdict(list)
    for r in rs:groups[r['terrain']].append(r)
    detail={}
    for terrain,rr in groups.items():
        y=np.array([r['target'] for r in rr]); truth=np.c_[y,y[:,0]-y[:,1]]
        measures={}
        for name in ('cargo','state','action'):
            a=np.array([r['predictions'][name] for r in rr]); e=np.c_[a,a[:,0]-a[:,1]]-truth
            measures[name]=dict(mae=np.abs(e).mean(0).tolist(),bias=e.mean(0).tolist(),over=(e>1e-8).sum(0).tolist(),
                under=(e< -1e-8).sum(0).tolist(),net_sign_accuracy=float(np.mean(np.sign(truth[:,-1])==np.sign(a[:,0]-a[:,1]))))
        detail[terrain]=dict(roots=len(rr),games=len(set(r['match'] for r in rr)),
            zero=[int(np.count_nonzero(truth[:,j]==0)) for j in range(3)],models=measures)
    summary={k:{m:np.mean([d['models'][k][m] for d in detail.values()],axis=0).tolist()
               for m in ('mae','bias','net_sign_accuracy')} for k in ('cargo','state','action')}
    gain=np.array([d['models']['action']['mae'][2]-d['models']['state']['mae'][2] for d in detail.values()])
    rng=np.random.RandomState(486731)
    return dict(maps=len(groups),roots=len(rs),games=len({r['match'] for r in rs}),summary=summary,by_map=detail,
        metrics_in_order=['own','enemy','own_minus_enemy'],mae_net_delta_state_95_map_bootstrap=np.quantile(rng.choice(gain,(10000,len(gain))).mean(1),[.025,.975]).tolist(),
        maps_net_improved_vs_state=int((gain<0).sum()), primary_better_than_both=all(summary['action']['mae'][2]<summary[k]['mae'][2] for k in ('cargo','state')))

def held_report(rs):
    reports={}
    for arm,name in [('challenger','v4'),('parent','v2')]:
        rows=[r for r in rs if r['source_arm']==arm]
        assert len(rows)==384
        reports[name]={section:report_held([r for r in rows if keep(r)]) for section,keep in {
           'all':lambda r:True,'late':lambda r:r['step']>=240,'multiple_loaded_enemies_near_base':lambda r:r['multi_enemy']}.items()}
    return reports

def evaluate_pair(row,w,inclusive):
    y=np.array(row['all_bank_actual'] if inclusive else row['actual'])
    preds=[predict(row['players'],row['seat'],row['step'],act,w) for act in row['orders']]
    pred={k:np.array([x[k] for x in preds]) for k in preds[0]}
    assert np.array_equal(pred['state'][0],pred['state'][1]) and np.array_equal(pred['cargo'][0],pred['cargo'][1])
    delta_y=(y[1]-y[0]);delta_pred=pred['action'][1]-pred['action'][0]
    score=float(delta_pred[0]-delta_pred[1]); b=int(score>1e-8)
    return dict(group=row['group'],seed=row['seed'],seat=row['seat'],terrain=row['terrain'],step=row['step'],orders=row['orders'],
                amount_arms=y.tolist(),predicted_arms={k:v.tolist() for k,v in pred.items()},delta_B_A_actual=delta_y.tolist(),
                delta_B_A_pred=delta_pred.tolist(),pred_net_difference=score,actual_net_difference=float(delta_y[0]-delta_y[1]),
                selected_B=bool(b),true_margin=row['margins'],official_change=float(b*(row['margins'][1]-row['margins'][0])))

def pair_report(rs,w,inclusive):
    outs=[evaluate_pair(r,w,inclusive) for r in rs if (not inclusive or r.get('all_bank_actual') is not None)]
    assert len({r['terrain'] for r in outs}&{r['terrain'] for r in read(H/'protocol.json')['train']})==0
    def mean_map(fun):
        groups=defaultdict(list)
        for r in outs:groups[r['terrain']].append(fun(r))
        return np.mean([np.mean(l,axis=0) for l in groups.values()],axis=0).tolist()
    positives=sum(r['selected_B'] and r['actual_net_difference']>1e-8 for r in outs)
    harms=sum(r['selected_B'] and r['actual_net_difference']< -1e-8 for r in outs)
    gain=mean_map(lambda r:r['selected_B']*r['actual_net_difference'])
    return dict(roots=outs,n=len(outs),maps=len({r['terrain'] for r in outs}),changes=sum(r['selected_B'] for r in outs),
       corrected=positives,harmed=harms,predicted_ties=sum(abs(r['pred_net_difference'])<=1e-8 for r in outs),
       true_net_ties=sum(abs(r['actual_net_difference'])<=1e-8 for r in outs),
       net_gain_vs_A= gain,official_gain_vs_A=mean_map(lambda r:r['official_change']),
       pair_gain_role=mean_map(lambda r: np.array(r['delta_B_A_actual'])*r['selected_B']),
       zero_action_delta_mae=mean_map(lambda r:abs(r['actual_net_difference'])),
       learned_action_delta_mae=mean_map(lambda r:abs(r['actual_net_difference']-r['pred_net_difference'])),
       action_gate=bool(positives>=2 and positives>harms and gain>0))

def exposed(w):
    import ryan_example
    out=[]
    for r in ryan_example.real_examples():
        brief={k:v for k,v in r.items() if k not in ('players','alternatives')};brief['predictions']=predict(r['players'],r['seat'],r['step'],r['order'],w)
        brief['alternatives']=[dict(**a,predictions=predict(r['players'],r['seat'],r['step'],a['order'],w)) for a in r['alternatives']]
        out.append(brief)
    return out

if __name__=='__main__':
    variant=sys.argv[1];assert variant in ('own','bank')
    bank=variant=='bank'
    prefix='bank_' if bank else ''
    w=read(H/(prefix+'weights.json'))
    p=H/(prefix+'held_predictions.json.gz')
    a=json.loads(gzip.decompress(p.read_bytes()))
    assert a['weights_sha256']==sha((H/(prefix+'weights.json')).read_bytes())
    rs=a['rows'];h=held_report(rs)
    dump(H/(prefix+'held_results.json'),dict(held=h,source=sha(p.read_bytes()),weights=a['weights_sha256']))
    import pairs
    pr=pair_report(pairs.rows(),w,bank)
    dump(H/(prefix+'paired_results.json'),dict(source=sha((H/'pairs.py').read_bytes()),weights=a['weights_sha256'],result=pr,
                exclusions=pairs.excluded(),gate_held_and_pairs=h['v4']['all']['primary_better_than_both'] and pr['action_gate']))
    if not bank:dump(H/'examples.json',exposed(w))
    for arm in ['v4','v2']:
        print(variant,arm,{k:v['mae'] for k,v in h[arm]['all']['summary'].items()},flush=True)
    print({k:v for k,v in pr.items() if k!='roots'},flush=True)
