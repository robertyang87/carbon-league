"""Reproduce all selected V4 ladder purchases at the public entry and at paid-v4 search work."""
import argparse,ast,collections,copy,gzip,json,os,sys,time,types
from pathlib import Path
from build import HERE,D,PARENT,sha
LOCK='e76396445c81a8770141f2c4e1dba2c8d34c69d95b03713d9a1e2e7f12f06fab'
MODES=('parent','candidate')


def load_mod(name,data):
    m=types.ModuleType(name);exec(compile(data,name,'exec'),m.__dict__);return m


def pub(frame,s):
    common=frame[0]['observation'];own=frame[s]['observation'];assert own['player']==s
    return dict(step=common['step'],player=s,players=common['players'],trees=common['trees'],carbon=own['carbon'],remainingOverageTime=own['remainingOverageTime'])


def put(path,j):
    raw=(json.dumps(j,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
    if path.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    tmp=path.with_suffix('.partial');tmp.write_bytes(raw);tmp.replace(path);return sha(raw)


def get_control():
    p=D/'endgame_cash_projection/response_check/run.py';tree=ast.parse(p.read_bytes());part=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('need','control_fn')],type_ignores=[])
    scope={'ast':ast};exec(compile(part,str(p),'exec'),scope);return scope['control_fn'],sha(p.read_bytes())


def read_raw(c,key,digest):
    b=(D/c[key]).read_bytes();assert sha(b)==c[digest];return json.loads(gzip.decompress(b))


def assess(obs,cfg,history,tr,model,coef):
    import numpy as np
    ps=obs['players'];s=obs['player'];t=obs['step'];own=ps[s];price=model._PLANT_LIB['quoted_price'](ps,s,cfg)
    occupied=set(y[0] for p in ps for y in p[3].values())|set(c for p in ps for c in p[1].values())
    out={};hist={lag:history[t-lag] for lag in (3,5) if t-lag in history}; changes=np.zeros((225,5),dtype=np.float32)
    q0=copy.deepcopy(tr);model._plant_reprice(obs,cfg,q0,changes);comp=np.zeros_like(changes)
    for uid,(cell,_,role) in own[2].items():
        if role!='PLANTER' or cell in occupied or price<=0:continue
        x=model._PLANT_LIB['features'](obs,uid,cfg,hist);z=[(v-m)/sd for v,m,sd in zip(x,coef['feature_mean'],coef['feature_scale'])];net=coef['means'][0]+sum(a*w[0] for a,w in zip(z,coef['coef']));bias=max(-1.,min(0.,net/price)) if own[0]>=price else 0.;comp[cell,0]=bias
        out[uid]=dict(cell=cell,predicted_net=net,price=price,public_cash_covers=own[0]>=price,logit_change=bias)
    assert np.array_equal(changes,comp)
    return out


def stable(q,base):
    return all(q[k]==base[k] for k in ('menu','issued','act0','opponent_samples','best','leaf_scores','depth_by_arm','native_aborted'))


def main(cpu):
    os.sched_setaffinity(0,{cpu});os.environ['CARBON_DEBUG']='1'
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    import numpy as np,torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(D/'opponent_policy/search_probe'))
    from full_search_check import root_call
    raw=(HERE/'protocol.json').read_bytes();assert sha(raw)==LOCK;prot=json.loads(raw);ident=prot['identity'];coef=json.loads((D/'tree_control_response/weights.json').read_bytes());cf,helper_sha=get_control()
    src={'parent':PARENT.read_bytes(),'candidate':(HERE/'submission.py').read_bytes()};assert sha(src['parent'])==ident['parent_sha256'] and sha(src['candidate'])==ident['candidate_sha256']
    m={n:load_mod(n,v) for n,v in src.items()};native={n:mod._search for n,mod in m.items()};control={n:cf(mod,src[n]) for n,mod in m.items()}
    for mod in m.values():assert mod._net() is not None and mod._SEARCH_OK and mod._S_ABORT==.58 and mod._SEED is None and mod._TEMP==1 and mod._S_MMAX==mod._S_M==1
    cost=list(m['parent']._S_COST);assert min(cost)>0
    freeze_forecast=(HERE.parent/'tree_forecast.json').read_bytes();forecast=json.loads(freeze_forecast);assert forecast['model_sha256']==ident['model_sha256'];rx=[g['teams'][0] for g in forecast['games'] if g['teams'][0]['side']=='v4'][0];rxrows={(v['step'],v['planter']):v for v in rx['rows']}
    results=[];counts=collections.Counter()
    for c in prot['games']:
        cp=HERE/(c['match']+'.roots.json.gz');r=read_raw(c,'archive','sha256');ledger=read_raw(c,'ledger','ledger_sha256'); side=c['seat']; cfg={k:v for k,v in r['configuration'].items() if k!='randomSeed'}
        frames=r['steps'];targets={v['step']:v for v in c['roots']};tr=m['parent'].BeliefTracker(m['parent']._Bel,side);prev=None;hist={};per=[]
        future_labels={(e['t'],e['planter']):e for e in ledger['planting'][0]['individual']}
        for t,frame in enumerate(frames[:-1]):
            obs=pub(frame,side);ps=obs['players'];assert obs['step']==t and 'full_carbon' not in obs
            if t in targets:
                target=targets[t];rng=target['rng']; pred=assess(obs,cfg,hist,tr,m['candidate'],coef);neg=[uid for uid,q in pred.items() if q['logit_change']<0.];o=MODES if (len(per)%2)==0 else MODES[::-1]
                measurements={name:root_call(m[name],tr,prev,obs,cfg,rng,cost) for name in o};pa,ca=measurements['parent'],measurements['candidate'];wchange=pa['issued']!=ca['issued']
                evaluated={'root_any_negative':bool(neg),'live_joint_changed':wchange,'matched_run':bool(neg or wchange),'matched_joint_changed':False,'reproduces_in_both':False}
                fixed={}
                if evaluated['matched_run']:
                    for name in o[::-1]:
                        mod=m[name];mod._response_cap=pa['candidates'];mod._response_depth=pa['depth'];mod._S_MMAX=pa['rollouts'];mod._search=control[name]
                        try:fixed[name]=root_call(mod,tr,prev,obs,cfg,rng,cost)
                        finally:mod._search=native[name];mod._S_MMAX=mod._S_M
                    x,y=fixed['parent'],fixed['candidate'];same_op=pa['opponent_samples']==x['opponent_samples']==y['opponent_samples']==ca['opponent_samples'];khr=('candidates','depth','rollouts','sorted_depths');same_work=all(x[k]==y[k]==pa[k] for k in khr)
                    baseline=stable(x,pa);repro_both=baseline and same_work and same_op and y['issued']==ca['issued'];eval_fixed=baseline and same_work and same_op and x['issued']!=y['issued']
                    evaluated.update(parent_restored=baseline,work_restored=same_work,equal_opponent=same_op,matched_joint_changed=eval_fixed,reproduces_in_both=eval_fixed and repro_both and wchange)
                purchases=[]
                for a in target['plants']:
                    uid=a['uid'];paid=future_labels[t,uid];predicted=pred[uid];assert paid['tree']==a['tree'] and frames[t+1][side]['action'].get(uid,'NONE')=='NONE'
                    assert abs(predicted['price']-paid['cost'])<1e-6
                    if c['name']=='RyanX':
                        fore=rxrows[t,uid];assert abs(predicted['predicted_net']-fore['predicted_net'])<1e-6
                    own_old,own_new=pa['issued'].get(uid,'NONE'),ca['issued'].get(uid,'NONE');b=(own_old=='NONE' and own_new!='NONE' and evaluated['reproduces_in_both'])
                    purchases.append(dict(**a,**{k:v for k,v in predicted.items() if k!='cell'},actual_first_tenure_net=paid['net'],parent_reconstructed=own_old,new_reconstructed=own_new,verified_stay_to_leave=b,verified_leave_to_stay=bool(own_old!='NONE' and own_new=='NONE' and evaluated['reproduces_in_both'])))
                per.append(dict(step=t,rng=rng,purchasers=purchases,all_planters=pred,check=evaluated,run_order=o,live=measurements,matched=fixed))
                if len(per)%12==0:print(c['name'],len(per),'/',len(targets),flush=True)
            history_view=m['candidate']._plant_history(tr,obs);hist[t]=ps
            if t-5 in hist:del hist[t-5]
            tr.step(ps,t,obs['carbon']);tr.set_actions(ps,frames[t+1][side]['action'] or {});prev=ps
        assert len(per)==len(targets);assert sum(len(v['purchasers']) for v in per)==len(future_labels);out=dict(game=c['name'],match=c['match'],seat=side,roots=per,protocol_sha256=LOCK)
        digest=put(cp,out);results.append((c,digest,per));print('GAME',c['name'],digest,flush=True)
    summarize(results,cpu,cost)


def summarize(results,cpu,cost):
    _,helper_sha=get_control(); freeze_forecast=(HERE.parent/'tree_forecast.json').read_bytes()
    summary=[]
    for c,digest,rs in results:
        plants=[p for r in rs for p in r['purchasers']];counters={k:sum(bool(r['check'][k]) for r in rs) for k in ('root_any_negative','live_joint_changed','matched_run','matched_joint_changed','reproduces_in_both')};q2={}
        for category,predicate in (('all',lambda p:True),('predicted_negative',lambda p:p['predicted_net']<0),('flagged',lambda p:p['logit_change']<0),('requires_new_cash',lambda p:not p['public_cash_covers']),('verified_stay_to_leave',lambda p:p['verified_stay_to_leave']),('verified_leave_to_stay',lambda p:p['verified_leave_to_stay'])):
            sel=[p for p in plants if predicate(p)];q2[category]=dict(n=len(sel),historical_loss=sum(p['actual_first_tenure_net']<0 for p in sel),historic_profit=sum(p['actual_first_tenure_net']>0 for p in sel),historic_net_sum=sum(p['actual_first_tenure_net'] for p in sel),original_reconstructed_stays=sum(p['parent_reconstructed']=='NONE' for p in sel),new_reconstructed_stays=sum(p['new_reconstructed']=='NONE' for p in sel))
        clocks={m:dict(calls=len(rs),max_wall=max(r['live'][m]['wall_s'] for r in rs),over_one=sum(r['live'][m]['wall_s']>1 for r in rs),zero_depth=sum(r['live'][m]['depth']==0 for r in rs)) for m in MODES};summary.append(dict(match=c['match'],opponent=c['name'],roots=len(rs),purchases=q2,checks=counters,clocks=clocks,roots_sha256=digest))
    put(HERE/'summary.json',dict(protocol_sha256=LOCK,source_sha256=sha(Path(__file__).read_bytes()),control_helper_sha256=helper_sha,original_forecast_sha256=sha(freeze_forecast),cpu=cpu,initial_native_cost=cost,games=summary));print(summary,flush=True)


def settle_existing(cpu):
    reg=(HERE/'protocol.json').read_bytes();assert sha(reg)==LOCK
    out=[]
    for c in json.loads(reg)['games']:
        p=HERE/(c['match']+'.roots.json.gz');b=p.read_bytes();r=json.loads(gzip.decompress(b))
        assert r['protocol_sha256']==LOCK and r['match']==c['match'] and [q['step'] for q in r['roots']]==[q['step'] for q in c['roots']]
        out.append((c,sha(b),r['roots']))
    # Live rows persist every completed call's ending costs. Historical startup calibration
    # was not written to the per-game records, so a resumed report must leave it unknown.
    summarize(out,cpu,None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu',type=int,required=True);p.add_argument('--settle-only',action='store_true');a=p.parse_args()
    (settle_existing if a.settle_only else main)(a.cpu)
