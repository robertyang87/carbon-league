"""Replay genuine archived V4 root searches, isolating changed opponent draws in successors."""
import argparse
import ast
import collections
import gzip
import hashlib
import json
import os
import sys
import types
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
LOCK='680374aefff6144eed1f702ad93910d6247b8f14843c005a1b8d1f4ce547d099'
sha=lambda b:hashlib.sha256(b).hexdigest()


def old_helpers():
    path=D/'root_rotation_policy/native/probe.py'
    node=ast.parse(path.read_bytes())
    code=ast.Module(body=[f for f in node.body if isinstance(f,ast.FunctionDef) and f.name in ('each_history','equal_work_fn')],type_ignores=[])
    scope=dict(ast=ast,zipfile=zipfile,json=json,sha=sha,gzip=gzip,D=D,CONTROL=D/'endgame_cash_projection/response_check/run.py')
    exec(compile(code,str(path),'exec'),scope)
    return scope['each_history'],scope['equal_work_fn']()


def same_order(a,b):
    return all(a[k]==b[k] for k in ('issued','act0','menu','opponent_samples','best','native_aborted','depth_by_arm','leaf_scores'))


def with_trace(mod,tr,previous,obs,cfg,rng,cost,call):
    import numpy as np
    from collections import defaultdict
    offset=obs['step'];side=obs['player'];logs=defaultdict(list);roots=[]
    old=mod._sample;orig_search=mod._search
    def sample(ps,who,lw,lb,generator,t,fixed=None):
        is_recorded=who!=side and t in (offset,offset+1)
        if is_recorded:
            # The hash is read before any sample. At depth one previous public players are the common root.
            rs=generator.get_state();seed=(rs[1].tobytes()+np.asarray(rs[2:4],'int64').tobytes()+np.asarray(rs[4],'float64').tobytes())
            token=sha(json.dumps(ps,separators=(',',':')).encode())+'/'+sha(seed)
            meta=dict(t=t,state_rng=token,lw=sha(lw.tobytes()),lb=sha(lb.tobytes()))
        result=old(ps,who,lw,lb,generator,t,fixed)
        if is_recorded:
            meta['order']=result;logs[t].append(meta)
        return result
    def watched_search(ob,cf,est,act0,lw,lb,prev,t0,home=()):
        roots.append([sha(lw.tobytes()),sha(lb.tobytes())])
        return orig_search(ob,cf,est,act0,lw,lb,prev,t0,home)
    mod._sample,mod._search=sample,watched_search
    try:r=call(mod,tr,previous,obs,cfg,rng,cost)
    finally:mod._sample,mod._search=old,orig_search
    assert len(roots)==1
    return dict(measurement=r,root_own_logits_sha=roots[0],root_enemy=logs[offset],successors=logs[offset+1])


def summary(ref,new,live):
    pa,ca=ref['measurement'],new['measurement']
    p,c=ref['successors'],new['successors']
    matched=[(a,b) for a,b in zip(p,c) if a['state_rng']==b['state_rng']]
    same=len(p)==len(c)==len(matched)
    if matched:
        assert all(a['lb']==b['lb'] for a,b in matched)
    changed=[i for i,(a,b) in enumerate(matched) if a['order']!=b['order']]
    changed_logits=[i for i,(a,b) in enumerate(matched) if a['lw']!=b['lw']]
    root_equal=pa['act0']==ca['act0'] and pa['opponent_samples']==ca['opponent_samples'] and ref['root_own_logits_sha']==new['root_own_logits_sha'] and ref['root_enemy']==new['root_enemy']
    khr=('candidates','depth','rollouts','sorted_depths')
    work=all(pa[k]==ca[k]==live['parent'][0][k] for k in khr) and pa['menu']==ca['menu']==live['parent'][0]['menu']
    live_parent_restored=same_order(pa,live['parent'][0]);live_diff=live['parent'][0]['issued']!=live['child'][0]['issued']
    repeats=all(same_order(x[0],x[1]) for x in live.values())
    live_agrees=ca['issued']==live['child'][0]['issued']
    return dict(parent_paid_work_restored=live_parent_restored,work_and_menu_match=work,root_views_actions_same=root_equal,
                first_step_all_states_rng_equal=same,first_step_matched=len(matched),first_step_worker_logits_changed=changed_logits,
                first_step_issued_enemy_changed=changed,live_diff=live_diff,each_package_live_repeats_exactly=repeats,
                controlled_changed=work and root_equal and same and pa['issued']!=ca['issued'],
                robust_changed=work and root_equal and same and live_parent_restored and repeats and live_agrees and live_diff)


def run(core, candidate_sha, gate_file):
    if len(candidate_sha)!=64:raise ValueError('Expected a frozen child SHA-256')
    # The decision file is written by the independent held-map evaluation.
    gate=json.loads(gate_file.read_bytes())
    if gate['domain']['v4']['pass_gate'] is not True:raise ValueError('No held-map PASS')
    os.sched_setaffinity(0,{core})
    for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(D/'opponent_policy/search_probe'))
    from full_search_check import root_call
    source={'parent':(D/'advantage_policy_update/v2_bc.py').read_bytes(),'child':(HERE.parent/'candidate.py').read_bytes()}
    p=(HERE/'protocol.json').read_bytes();assert sha(p)==LOCK;cfgreg=json.loads(p)
    assert sha(source['parent'])==cfgreg['parent'] and sha(source['child'])==candidate_sha
    each_history,equal_work=old_helpers();mods={};normal={};controlled={}
    for name,src in source.items():
        mod=types.ModuleType(name);exec(compile(src,name,'exec'),mod.__dict__);mods[name]=mod
        assert mod._net() is not None and mod._SEARCH_OK and mod._S_ABORT==.58 and mod._TEMP==1
        normal[name]=mod._search;controlled[name]=equal_work(mod,src)
    s=D.parents[1]/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json'
    cfg={k:v['default'] for k,v in json.loads(s.read_bytes())['configuration'].items() if isinstance(v,dict) and 'default' in v}|{'episodeSteps':300}
    initial=list(mods['parent']._S_COST);rows=[]
    for ticket in cfgreg['tickets']:
        seat=ticket['seat'];t=ticket['step'];rng=ticket['rng']
        ref=mods['parent'];belief=ref.BeliefTracker(ref._Bel,seat);prev=None
        for obs,prior,vis in each_history(ticket):
            k=obs['step']
            if k==t-1:cost=root_call(ref,belief,prev,obs,cfg,rng^0x591141b,initial)['ending_costs']
            if k==t:break
            belief.step(obs['players'],k,obs['carbon']);belief.set_actions(obs['players'],prior);prev=obs['players']
        assert k==t
        live={'parent':[],'child':[]};seq=('parent','child','child','parent') if len(rows)%2==0 else ('child','parent','parent','child')
        for name in seq:live[name].append(root_call(mods[name],belief,prev,obs,cfg,rng,cost))
        par=live['parent'][0];work=[par[k] for k in ('candidates','depth','rollouts')];fixed={}
        if min(work)>0:
            for name in (seq[0],seq[1]):
                mod=mods[name];mod._response_cap,mod._response_depth,mod._S_MMAX=work;mod._search=controlled[name]
                try:fixed[name]=with_trace(mod,belief,prev,obs,cfg,rng,cost,root_call)
                finally:mod._S_MMAX=mod._S_M;mod._search=normal[name]
            check=summary(fixed['parent'],fixed['child'],live)
        else:check={'insufficient_paid_work':True}
        rows.append(dict(ticket=ticket,public_sha256=vis,live=live,fixed_work=fixed,metrics=check))
    obj=dict(protocol_sha256=LOCK,gate_sha256=sha(gate_file.read_bytes()),code_sha256=sha(Path(__file__).read_bytes()),parent=sha(source['parent']),candidate=candidate_sha,rows=rows)
    (HERE/'outcomes.json.gz').write_bytes(gzip.compress((json.dumps(obj,sort_keys=True,allow_nan=False)+'\n').encode(),mtime=0))
    print([r['metrics'] for r in rows])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--core',type=int,default=3);ap.add_argument('--candidate-sha',required=True);ap.add_argument('--gate-file',required=True,type=Path)
    args=ap.parse_args();run(args.core,args.candidate_sha,args.gate_file)
