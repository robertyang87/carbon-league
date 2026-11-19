"""Fit an antisymmetric shared action utility and audit choices on held boards and a later sampler."""
from collections import Counter,defaultdict
import numpy as np
from policy import choose
from collect import HERE,SIGNED,read,sha,write

TARGETS=['deposit_margin_24','deposit_margin_terminal','official_margin_terminal']
def weights(rows):
    sizes=Counter(r['terrain'] for r in rows)
    return np.array([1/(len(sizes)*sizes[r['terrain']]) for r in rows])
def target(rows):return np.array([r['target'] for r in rows])
def bounds(x):return np.quantile(x,[.025,.5,.975],axis=0).round(3).tolist()
def boardmean(rows,v):return np.average(v,axis=0,weights=weights(rows))
def indices(rows,col):return [r[col] for r in rows]
def fit(rows,X,shuffle=None):
    w=weights(rows)
    b=X[indices(rows,'menu_start')];a=X[indices(rows,'alternative')];y=target(rows).copy()
    if shuffle is not None:
        groups=defaultdict(list);permutation=np.random.default_rng(shuffle)
        for i,r in enumerate(rows):groups[r['family'],r['role']].append(i)
        for g in groups.values():y[g]=y[permutation.permutation(g)]
    mu=np.average((a+b)/2,axis=0,weights=w)
    sd=np.maximum(.1,np.sqrt(np.average(((a-mu)**2+(b-mu)**2)/2,axis=0,weights=w)))
    sy=np.maximum(1,np.sqrt(np.average(y*y,axis=0,weights=w)))
    a,b=(a-mu)/sd,(b-mu)/sd
    y=np.clip(y/sy,-2,2)
    rng=np.random.default_rng(62471)
    pars=[rng.normal(0,.18,(a.shape[1],4)),np.zeros(4),rng.normal(0,.18,(4,3))]
    first=[np.zeros_like(p) for p in pars];second=[p.copy() for p in first]
    losses=[]
    for t in range(1,221):
        k,h,o=pars
        ua,ub=np.tanh(a@k+h),np.tanh(b@k+h)
        d=ua-ub
        e=d@o-y
        err=2/3*e*w[:,None]
        ea=(err@o.T)*(1-ua**2);eb=-(err@o.T)*(1-ub**2)
        grads=[a.T@ea+b.T@eb+.05*k,(ea+eb).sum(axis=0),d.T@err+.05*o]
        if t in (1,220):losses.append(float(np.average((e*e).mean(axis=1),weights=w)+.025*((k*k).sum()+(o*o).sum())))
        for p,g,m,v in zip(pars,grads,first,second,strict=True):
            m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
            p-=.012*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1.e-8)
    return dict(mu=mu,sd=sd,target_scale=sy,in_weight=pars[0],bias=pars[1],out_weight=pars[2]),losses
def score(X,m):
    return np.tanh((X-m['mu'])/m['sd']@m['in_weight']+m['bias'])@m['out_weight']
def matched_predict(train,r):
    sel=[x for x in train if x['family']==r['family'] and x['role']==r['role']]
    if not sel:sel=[x for x in train if x['role']==r['role']]
    return boardmean(sel,target(sel)) if sel else np.zeros(3)
def decisions(rows,X,m,train):
    scores=score(X,m)
    res=[]
    for r in rows:
        lo,idx=r['menu_start'],r['alternative']
        pair=scores[[lo,idx]]
        change=choose(pair)
        length=r['free_menu_length']
        free=lo+choose(scores[lo:lo+length]) if length is not None else None
        exact='no_full_menu' if free is None else ('parent' if free==lo else 'alternative' if free==idx else 'unmeasured')
        matched=matched_predict(train,r)
        res.append({k:r[k] for k in ('case','identity','seed','terrain','seat','t','role','family','split','target')}|
                   dict(predicted_margin=(np.diff(pair,axis=0)[0]*m['target_scale']).round(6).tolist(),
                        signed_choice=int(change),matched_choice=int(all(matched>0)),
                        matched_prediction=matched.round(6).tolist(),full_menu_choice=exact,
                        free_menu_delta_index=free-lo if free is not None else None))
    return res
def aggregate(rows):
    if not rows:return {}
    y=target(rows)
    hits=np.array([r['signed_choice'] for r in rows]);matched=np.array([r['matched_choice'] for r in rows])
    vals={'learned':y*hits[:,None],'matched_sampler_role':y*matched[:,None],'always_propose':y,
          'keep_parent':y*0,'learned_minus_matched':y*(hits-matched)[:,None]}
    mapvals=defaultdict(dict);rng=np.random.default_rng(661147)
    for terr in sorted(set(indices(rows,'terrain'))):
        mask=np.array([r['terrain']==terr for r in rows])
        for k,value in vals.items():mapvals[terr][k]=value[mask].mean(axis=0).round(6).tolist()
    bs=rng.integers(0,len(mapvals),(2000,len(mapvals)))
    result=dict(roots=len(rows),terrains=len(mapvals),cases_changed=int(hits.sum()),matched_changes=int(matched.sum()),
                positive_terrains_per_target=np.sum([np.array(v['learned'])>0 for v in mapvals.values()],axis=0).tolist(),
                mean={k:boardmean(rows,x).round(4).tolist() for k,x in vals.items()},
                descriptive_board_bootstrap={k:bounds(np.array([v[k] for v in mapvals.values()])[bs].mean(axis=1)) for k in ['learned','learned_minus_matched']},
                full_menu=dict(Counter(indices(rows,'full_menu_choice'))),maps=dict(mapvals))
    result['direction_accuracy_nonzero']=[]
    pred=np.array(indices(rows,'predicted_margin'))
    for j in range(3):
        mask=y[:,j]!=0
        result['direction_accuracy_nonzero'].append(dict(correct=int(np.sum(np.sign(y[mask,j])==np.sign(pred[mask,j]))),den=int(mask.sum())))
    return result
def repeatability(rr,kind):
    parent={r['identity']:r for r in rr if r['split'] in ('held','train')};rows=[]
    for r in rr:
        if r['split']!=kind:continue
        p=parent[r['identity']]
        assert p['both_commands']==r['both_commands'] and p['public_sha256']==r['public_sha256']
        a,b=np.array(p['target']),np.array(r['target'])
        rows.append(dict(seed=r['seed'],seat=r['seat'],t=r['t'],identity=r['identity'],
                         primary=a.tolist(),redraw=b.tolist(),sign_flip=(a*b<0).tolist(),case=r['case']))
    by_split={}
    for g in ('train','held'):
        group=[r for r in rows if parent[r['identity']]['split']==g]
        by_split[g]={'redraws':len(group),'roots':len(set(indices(group,'identity'))),
                     'same_sign':np.sum([np.sign(r['primary'])==np.sign(r['redraw']) for r in group],axis=0).tolist() if group else [0,0,0]}
    return dict(draws=len(rows),unique_roots=len(set(indices(rows,'identity'))),flips=np.sum(indices(rows,'sign_flip'),axis=0).tolist(),by_primary_split=by_split,rows=rows)

def additional_menu_matches(rr,dec):
    records=read(SIGNED/'native_inputs.json.gz')['rows']
    menus={r['case']:r['menu_commands'] for r in records}
    chosen={r['case']:r for r in dec['held']+dec['train']}
    extra=[]
    for r in rr:
        if r['split']!='later_option':continue
        hits=[q for q in rr if q['split'] in ('train','held') and q['public_sha256']==r['public_sha256']
              and q['source_sha256']==r['source_sha256'] and q['both_commands'][0]==r['both_commands'][0]]
        for q in hits:
            row=chosen[q['case']];commands=menus[q['case']][row['free_menu_delta_index']]
            extra.append(dict(primary=q['case'],supplement=r['case'],split=q['split'],original_free_menu_choice=row['full_menu_choice'],
                              free_menu_command=commands,observed_additional_command=r['both_commands'][1],
                              measured_third_is_free_choice=commands==r['both_commands'][1],additional_target=r['target']))
    return extra
def main():
    data=read(HERE/'data.json');assert data['rows']==sha(HERE/'rows.json.gz') and data['feature_file']==sha(HERE/'public_features.npz')
    rr=read(HERE/'rows.json.gz');X=np.load(HERE/'public_features.npz')['features']
    groups={s:[r for r in rr if r['split']==s] for s in ('train','held','transfer')}
    train=groups['train']
    for a,b in [('train','held'),('train','transfer'),('held','transfer')]:
        assert set(indices(groups[a],'terrain')).isdisjoint(indices(groups[b],'terrain'))
    m,losses=fit(train,X)
    np.savez_compressed(HERE/'weights.npz',**m)
    dec={s:decisions(rows,X,m,train) for s,rows in groups.items()}
    devloo=[]
    for terr in sorted(set(indices(train,'terrain'))):
        tr=[r for r in train if r['terrain']!=terr];omit=[r for r in train if r['terrain']==terr]
        omitted,_=fit(tr,X);devloo.extend(decisions(omit,X,omitted,tr))
    dec['train_board_crossfit']=devloo
    totals={s:aggregate(rows) for s,rows in dec.items()}
    null=defaultdict(list)
    for sh in range(32):
        mm,_=fit(train,X,11051+sh)
        for s in ('held','transfer'):
            records=decisions(groups[s],X,mm,train)
            null[s].append(boardmean(records,target(records)*np.array(indices(records,'signed_choice'))[:,None]).tolist())
    sanity={}
    for s,rows in dec.items():
        sanity[s]={k:{str(v):aggregate([r for r in rows if r[k]==v]) for v in sorted(set(indices(rows,k)))} for k in ('role','family','seat')}
    held=totals['held'];gain=np.asarray(held['mean']['learned'])
    matched_adv=np.array(held['mean']['learned_minus_matched'])
    rrh=dec['held']
    checks=dict(positive_official_and_terminal_deposit=bool(min(gain[1:])>0),
                exceeds_matched_on_both=bool(min(matched_adv[1:])>0),
                changed_both_worker_roles=all(any(r['role']==role and r['signed_choice'] for r in rrh) for role in ['collector','planter']),
                more_than_half_free_menu_measured=2*(held['full_menu'].get('parent',0)+held['full_menu'].get('alternative',0))>len(rrh))
    write(HERE/'decisions.json.gz',dec,True)
    write(HERE/'results.json',dict(target_order=TARGETS,fit_data=sha(HERE/'data.json'),weights=sha(HERE/'weights.npz'),
          decisions=sha(HERE/'decisions.json.gz'),optimized_parameters=sum(m[x].size for x in ['in_weight','bias','out_weight']),
          loss_first_last=losses,board_summary=totals,strata=sanity,repeatability=repeatability(rr,'repeat'),
          shuffled_train_labels={s:{'num_fits':len(v),'means_by_fit':v,'quantiles':bounds(v)} for s,v in null.items()},
          later_v1_repeatability=repeatability(rr,'later_repeat'),later_v1_options=decisions([r for r in rr if r['split']=='later_option'],X,m,train),
          third_menu_reconciliation=additional_menu_matches(rr,dec),checks=checks,status='OFFLINE_ELIGIBLE' if all(checks.values()) else 'STOP'))
    for s in dec:print(s,totals[s]['roots'],totals[s]['terrains'],totals[s]['mean'],totals[s]['full_menu'])
    print(checks)
if __name__=='__main__':main()
