"""Ridge first-tenure net, with all M3-family replay games excluded from every split."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
OLD=BASE/'planter_routes_dev'
sys.path.insert(0,str(OLD))
import train
from features import _ROUTE_DEST, _ROUTE_FIELDS, PlanterRouteState
sys.path.insert(0,str(BASE))
from turning_point_ledger import FastCarbon,CashEvents

def h(data):
    return hashlib.sha256(data).hexdigest()

def save(name, obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n')

def banned(name):
    s=''.join(ch for ch in name.casefold() if ch.isalnum())
    return any(token in s for token in ('master3','m3s','3s2','d679ed45ab595e8b7bb1ea51c3451154'))

def clean(g):
    return not any(banned(n) for n in g['seat_names']+[i for i in g['seat_strategy_ids'] if i])

def select():
    old=json.loads((OLD/'split.json').read_bytes());manifest=json.loads((BASE/'all_top5/manifest.json').read_bytes())
    discarded={k:[g for g in gs if not clean(g)] for k,gs in old.items()}
    new={k:[g for g in gs if clean(g)] for k,gs in old.items()}
    seedset={g['seed'] for gs in old.values() for g in gs}
    seedset.update(g['seed'] for g in json.loads((BASE/'latest5_inventory.json').read_bytes())['games'])
    seedset.add(621456920)
    source=sorted(manifest['games'],key=lambda g:train.sha('route-map-1|'+str(g['seed'])+g['id']))
    substitutions={k:[] for k in old}
    for fold, slots in old.items():
        if len(new[fold])==len(slots):
            continue
        if fold.startswith('test'):
            raise ValueError('the original independent test must stay immutable')
        for g in source:
            if len(new[fold])>=len(slots):break
            if g['seed'] in seedset or not clean(g):continue
            ext=[n for n,i in zip(g['seat_names'],g['seat_strategy_ids']) if i is None]
            if any(int(train.sha('route-opponent-1|'+n),16)%5==0 for n in ext):continue
            x={k:g[k] for k in slots[0]}
            new[fold].append(x);substitutions[fold].append(x);seedset.add(g['seed'])
    included=[g for gs in new.values() for g in gs]
    if len(included)!=62 or len({g['seed'] for g in included})!=62 or not all(clean(g) for g in included):
        raise ValueError('unexpected split or opponent')
    return new,discarded,substitutions

def attestation(g):
    path=train.CACHE/g['path'];data=path.read_bytes()
    if h(data)!=g['sha256']:raise ValueError('wrong replay')
    work=train.CACHE/'plant_work'/(g['id']+'.gz');record=json.loads(gzip.decompress(work.read_bytes()))
    raw=json.loads(gzip.decompress(data));cfg=raw['configuration'];fs=raw['steps']
    sim=FastCarbon(cfg).load_observation(fs[0][0]['observation']);sim.cash=CashEvents(sim.cash)
    ev={}
    for t in range(len(fs)-1):
        sim.cash.events.clear()
        sim.step(*[a['action'] or {} for a in fs[t+1]])
        for row in sim.cash.events:
            if row[0] in ('plant','seize'):
                key=(t,row[1],row[3],row[4])
                if key in ev:raise ValueError(key)
                ev[key]=[row[0],-row[2]]
    cases=[]
    for a in record['purchases']:
        key=(a[0],a[1],a[2],a[4]);kind,cost=ev.pop(key)
        if kind!=a[5] or abs(cost-a[6])>1e-4:raise ValueError((key,cost,a))
        before=a[0]-1
        if not 45<=before<196:continue
        previous=fs[before][0]['observation']['players'][a[1]][2].get(a[2]);move=fs[before+1][a[1]]['action'].get(a[2])
        if previous is None or move not in train.MOVES or int(_ROUTE_DEST[previous[0],train.MOVES[move]])!=a[4]:continue
        now=fs[before][0]['observation'];st=PlanterRouteState()
        st.observe(now,a[1],fs[before][a[1]]['observation']['carbon'],cfg,before)
        pred=st.features(a[4])
        if pred is None:continue
        cases.append({'pre_move_t':before,'payment_action_t':a[0],'payment_post_t':a[7],'seat':a[1],'uid':a[2],'cell':a[4],
                      'kind':kind,'root_quote':pred[1],'post_move_actual_cost':cost,
                      'first_tenure_tree_income':a[11],'first_tenure_tree_collision_cash':a[13],'label':a[11]+a[13]-a[6]})
    if ev:raise ValueError('missing label payment')
    diffs=[abs(r['root_quote']-r['post_move_actual_cost']) for r in cases]
    samples=[]
    if cases:
        samples=[cases[0],cases[int(np.argmax(diffs))]]
    return {'id':g['id'],'all_payments_checked':len(record['purchases']),'observed_move_and_paid_rows':len(cases),
            'root_quote_differs_over_cent':sum(v>.01 for v in diffs),'max_quote_delta':max(diffs,default=0.),
            'examples':samples}

def measure(ds,model,anchor):
    x,y,values,w,ident=ds
    yes=(y==1)
    ids=np.array(ident)[yes];values=values[yes];w=w[yes];w=w/w.sum()
    pred=np.clip(((x[yes]-model['mean'])/model['scale'])@model['net_weights']+model['net_bias'],-150,250)
    def mean(q,mask):return float(w[mask]@q[mask]/sum(w[mask])) if mask.any() else None
    def corr(a,b,p):
        a=a-p@a;b=b-p@b
        denom=np.sqrt((p@a**2)*(p@b**2))
        return float(p@(a*b)/denom) if denom else None
    def group(mask):
        return {'purchases':int(mask.sum()),'mean_predicted':mean(pred,mask),'mean_realized':mean(values,mask),'positive_actual_fraction':mean(values>0,mask)}
    # Heldout rankings compare two purchases on the same terrain. No cross-map resampling of rows.
    within=[]
    for uid in sorted(set(ids)):
        ok=ids==uid
        x1,x2=pred[ok],values[ok]
        ix,iy=np.triu_indices(len(x1),1)
        dx=x1[ix]-x1[iy]; dy=x2[ix]-x2[iy];valid=dy!=0
        if valid.any():
            agree=(dx[valid]*dy[valid]>0)+.5*(dx[valid]==0)
            within.append({'id':uid,'purchases':len(x1),'pairs':int(valid.sum()),'order_accuracy':float(agree.mean())})
    q25,q75=np.quantile(pred,[.25,.75])
    if q25>=q75:
        raise ValueError('collapsed conditional forecast quartiles')
    return {'conditional_quartiles':{'definition':'unweighted linear empirical 25th/75th percentile on actual paid rows, inclusive ties; group means retain fitted map/move weights reconditioned on paid rows',
                                     'q25':float(q25),'q75':float(q75),
                                     'bottom':group(pred<=q25),'top':group(pred>=q75)},
            'maps':len(set(ident)),'all_moves':len(y),'paid':len(values),'net_mean':float(w@values),'mae':float(w@abs(values-pred)),
            'mae_constant_paid_mean':float(w@abs(values-anchor)),'rmse':float(np.sqrt(w@(values-pred)**2)),
            'rmse_constant_paid_mean':float(np.sqrt(w@(values-anchor)**2)),'weighted_rank_correlation':corr(rankdata(pred),rankdata(values),w),
            'positive_prediction':group(pred>0),'nonpositive_prediction':group(pred<=0),'ranking_within_maps':within,
            'equal_map_pair_order_accuracy':float(np.mean([z['order_accuracy'] for z in within]))}

def main():
    split,drops,subs=select()
    save('value_split.json',{'game_lists':split,'removed_whole_games':drops,'replacement_games':subs,
                            'old_split_sha256':h((OLD/'split.json').read_bytes()),'source_manifest_sha256':h((BASE/'all_top5/manifest.json').read_bytes())})
    ds=train.pack(split['train']);model=train.fit(ds)
    anchor=model['net_bias']*0.0+float(np.average(ds[2][ds[1]==1],weights=ds[3][ds[1]==1]))
    save('value_weights.json',model)
    print('weights ready',len(ds[1]),int(ds[1].sum()),flush=True)
    measures={}
    for name,games in split.items():
        measures[name]=measure(ds if name=='train' else train.pack(games),model,anchor)
    selected=[split[k][0] for k in split]
    checks=[attestation(g) for g in selected]
    gates={k:v['conditional_quartiles']['top']['mean_realized']>v['conditional_quartiles']['bottom']['mean_realized']
           for k,v in measures.items() if k.startswith('test_')}
    save('value_evaluation.json',{'top_above_bottom_both_untouched_test_groups':all(gates.values()),'top_above_bottom_by_test_group':gates,'weights_sha256':h((HERE/'value_weights.json').read_bytes()),'model_source_sha256':h((OLD/'train.py').read_bytes()),
         'features_sha256':h((OLD/'features.py').read_bytes()),'fields':list(_ROUTE_FIELDS),'constant_training_paid_net':anchor,
         'conditional_measures':measures,'actual_payments_checked':checks})

if __name__=='__main__':
    main()
