"""Verify exact old and deployed-new full-command likelihood, separate terrain-held projection accuracy from fit."""
from collections import Counter,defaultdict
import hashlib
import importlib.util
import numpy as np
from common import HERE, ROOT, assert_cpu, net, policy, read, sha, write
spec=importlib.util.spec_from_file_location('balanced_training_provider', HERE/'run.py')
training=importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
context=training.context
from decoder import instrument
from conditioned import Conditioning


def direct(root,slot):
    branches=[];boundaries={0.,1.}
    for residual in (False,True):
        _,details=root.inspect(residual,slot);p,h=details[slot]
        cuts=np.cumsum(p);cuts[-1]=1.;boundaries.update(map(float,cuts[:-1]))
        y={j:root.inspect(residual,slot,int(j))[0] for j in np.flatnonzero(p)}
        branches.append((cuts,y))
    law=defaultdict(float);intervals=[]
    ends=sorted(v for v in boundaries if 0<=v<=1)
    for lo,hi in zip(ends,ends[1:]):
        labels=[pool[min(4,int(np.searchsorted(cuts,lo,'right')))] for cuts,pool in branches]
        y,guard=policy.guard(root.m,root.x,*labels);k=policy.key(root.f.to_actions(y))
        law[k]+=hi-lo;intervals.append((lo,hi,k,guard))
    return dict(law),intervals


def check_public_projection():
    with np.load(HERE/'projection_inputs.npz') as dat:h,y,ids=dat['h'],dat['y'],dat['terrain']
    out=[]
    for terrain in sorted(set(ids)):
        train=ids!=terrain;proj=Conditioning.fit(h[train],y[train],ids[train].tolist())
        predicted=proj.features(h[~train]);t=y[~train]
        physical=np.c_[np.log1p(t[:,0]),t[:,1]/14.,t[:,2]];z=(physical-proj.target_mean)/proj.target_scale
        out.append(dict(terrain=terrain,slots=len(z),mse=np.mean((z-predicted)**2,axis=0).tolist(),constant_mse=np.mean(z*z,axis=0).tolist()))
    return out


def main():
    assert_cpu();schedule=read(HERE/'maps.json');p=read(HERE/'package.json');u=read(HERE/'update.json')
    assert sha(HERE/'candidate.py')==p['candidate_sha256']
    frozen=instrument(net('it20_self'));student=instrument(policy.load(HERE/'candidate.py'))
    tested={};rows=[]
    if u['decision']=='UPDATE':
        for _,data in u['trials'][-1]['maps'].items():
            for r in data['roots']:tested[r['seed'],r['seat'],r['t']]=r['new_support']
    for source in sorted((HERE/'train').glob('*.json.gz')):
        sample=read(source)
        if sample.get('skipped'):continue
        st=sample['state'];r=st['header'];slot=r['choice']['slot'];oldroot,*_=context(frozen,r['seed'],r['original_seat'],r['t'])
        newroot,*_=context(student,r['seed'],r['original_seat'],r['t'])
        old,intervals=direct(oldroot,slot);new,_=direct(newroot,slot);ref=st['executed_mass']
        assert set(old)==set(ref) and max(abs(old[k]-ref[k]) for k in old)<1e-12
        expected=tested.get((r['seed'],r['seat'],r['t']),ref)
        assert set(new)==set(expected) and max(abs(new[k]-expected[k]) for k in new)<1e-12
        guard=off=0
        for s in st['samples']:
            lo,hi,key,rolled=next((a,b,k,boo) for a,b,k,boo in intervals if a<=s['common_uniform']<b)
            assert key==s['key'] and rolled==s['rollback'] and old[key]==s['old_p']
            assert sample['outcomes'][key]['actions'][0][r['seat']]==__import__('json').loads(key)
            guard+=rolled;off+=key!=r['choice']['greedy']
        rows.append(dict(seed=r['seed'],seat=r['seat'],t=r['t'],slot=slot,worker_uid=r['choice']['uid'],
               cargo_distance_threat=r['choice']['context'],samples=len(st['samples']),guarded=guard,num_nongreedy=off,
               distinct_native_endings=len(sample['outcomes']),old_support_count=len(old),new_support_count=len(new),
               old_new_total_variation=.5*sum(abs(old.get(k,0)-new.get(k,0)) for k in old.keys()|new.keys()),
               effect_official_u_range=np.ptp([v['official']['u_by_seat'][r['seat']] for v in sample['outcomes'].values()]).item()))
    write(HERE/'likelihood_audit.json',dict(candidate_sha256=p['candidate_sha256'],row_count=len(rows),rows=rows,
          checked_draws=sum(r['samples'] for r in rows),guarded_draws=sum(r['guarded'] for r in rows),
          nongreedy_draws=sum(r['num_nongreedy'] for r in rows),distinct_endings=sum(r['distinct_native_endings'] for r in rows),
          projection_leave_one_training_terrain_out=check_public_projection()))
    print('verified direct joint probabilities',len(rows),sum(r['samples'] for r in rows),flush=True)


if __name__=='__main__':main()
