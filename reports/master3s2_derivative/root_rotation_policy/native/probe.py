"""Measure the frozen two-view policy on archived, preselected official public requests."""
import argparse
import ast
import collections
import gzip
import json
import os
from pathlib import Path
import sys
import time
import types
import zipfile

from register import HERE,D,PARENT,CANDIDATE,sha

LOCK='e36893f8eb94d8ba14b2edc5fc47f7fcaa36d34b663a7938b95ac3305a443a3e'
CONTROL=D/'endgame_cash_projection/response_check/run.py'
BASE=D/'advantage_policy_update/v2_bc.py'
NEW=HERE.parent/'submission.py'
MODES=('parent','candidate')


def save(path,data):
    raw=(json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    tmp=path.with_suffix('.partial');tmp.write_bytes(raw);tmp.replace(path)


def equal_work_fn():
    tree=ast.parse(CONTROL.read_bytes());sel=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('need','control_fn')]
    space={'ast':ast};exec(compile(ast.Module(body=sel,type_ignores=[]),str(CONTROL),'exec'),space);return space['control_fn']


def each_history(ticket):
    path=D/ticket['archive'];assert sha(path.read_bytes())==ticket['archive_sha256'];prefix=ticket['arm']+'/'
    with zipfile.ZipFile(path) as z:
        reg=z.read(prefix+'registration.json');assert sha(reg)==ticket['registration_sha256']
        receipt=json.loads(z.read(prefix+'receipt.json'));assert sha(reg)==receipt['protocol_sha256']
        raw=z.read(prefix+'turns.jsonl');assert sha(raw)==receipt['turns_sha256']
        obs=gzip.decompress(z.read(prefix+'observations.jsonl.gz')); turns=list(map(json.loads,raw.splitlines())); frames=list(map(json.loads,obs.splitlines()))
    assert len(frames)==len(turns)==receipt['official_transitions']==299 and receipt['counted']
    seat=ticket['seat']
    for i,(turn,frame) in enumerate(zip(turns,frames,strict=True)):
        ob=frame['observations'][seat];assert ob['step']==i==turn['step'] and ob['player']==seat and 'full_carbon' not in ob
        checksum=sha((json.dumps(ob,separators=(',',':'))+'\n').encode());assert checksum==frame['public_observation_sha256'][seat]==turn['public_observation_sha256'][seat]
        yield ob,turn['calls'][seat]['response']['action'],checksum


def monitor(fn,durations):
    def watched(*args,**kw):
        start=time.perf_counter();result=fn(*args,**kw);durations.append(time.perf_counter()-start);return result
    return watched


def experiment(core):
    os.sched_setaffinity(0,{core});os.environ['CARBON_DEBUG']='1'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(D/'opponent_policy/search_probe'))
    from full_search_check import root_call
    raw=(HERE/'protocol.json').read_bytes();assert sha(raw)==LOCK;prot=json.loads(raw)
    src={'parent':BASE.read_bytes(),'candidate':NEW.read_bytes()};assert sha(src['parent'])==PARENT and sha(src['candidate'])==CANDIDATE
    eq=equal_work_fn();mods={};control={};native={};durations={m:[] for m in MODES}
    for tag in MODES:
        m=types.ModuleType('rr_'+tag);exec(compile(src[tag],tag,'exec'),m.__dict__);mods[tag]=m
        assert m._net() is not None and m._S_ABORT==.58 and m._TEMP==1 and m._SEED is None and m._SEARCH_OK
        native[tag]=m._search;control[tag]=eq(m,src[tag]);stage='forward' if tag=='parent' else '_root_rotation_policy';where=m._POLICY if tag=='parent' else m
        setattr(where,stage,monitor(getattr(where,stage),durations[tag]))
    assert mods['parent']._S_M==mods['candidate']._S_M==mods['parent']._S_MMAX==mods['candidate']._S_MMAX==1
    schema=json.loads((D.parents[1]/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json').read_bytes())['configuration']
    cfg={k:v['default'] for k,v in schema.items() if isinstance(v,dict) and 'default' in v}|{'episodeSteps':300}
    opening=list(mods['parent']._S_COST);assert np.isfinite(opening).all() and min(opening)>0
    rows=[]; hits=collections.defaultdict(list)
    for c in prot['tickets']:hits[c['archive']].append(c)
    for tickets in hits.values():
        c=tickets[0]; targets={r['step']:r for r in tickets}; seat=c['seat'];tr=mods['parent'].BeliefTracker(mods['parent']._Bel,seat);prev=None
        for ob,old,checksum in each_history(c):
            i=ob['step']
            if i+1 in targets:
                warm=root_call(mods['parent'],tr,prev,ob,cfg,targets[i+1]['rng']^0x12f191ac,opening)
            if i in targets:
                ticket=targets[i];rng=ticket['rng'];startcost=warm['ending_costs'];row=dict(ticket=ticket,visible_before_sha256=checksum,warm_wall_s=warm['wall_s'],warm_cost=startcost)
                o=('parent','candidate','candidate','parent') if len(rows)%2==0 else ('candidate','parent','parent','candidate');row['order']=o
                for tier in ('live','equal_work'):
                    out={name:[] for name in MODES}
                    for name in o:
                        m=mods[name];m._S_MMAX=m._S_M
                        if tier=='equal_work':
                            m._response_cap,m._response_depth,m._S_MMAX=k,h,r
                            m._search=control[name]
                        durations[name].clear()
                        try:q=root_call(m,tr,prev,ob,cfg,rng,startcost)
                        finally:m._search=native[name];m._S_MMAX=m._S_M
                        assert len(durations[name])==1
                        q['root_policy_wall_s']=durations[name][0];out[name].append(q)
                    row[tier]=out
                    if tier=='live':
                        case=out['parent'][0];k,h,r=case['candidates'],case['depth'],case['rollouts']
                live=row['live'];cw=row['equal_work'];strict=('act0','issued','menu','depth_by_arm','leaf_scores','opponent_samples','best','native_aborted')
                invariant=lambda rs,key:all(z[key]==rs[0][key] for z in rs)
                parent_restored=all(x[key]==live['parent'][0][key] for x in cw['parent'] for key in strict)
                exact_work=all((q['candidates'],q['depth'],q['rollouts'])==(k,h,r) and q['sorted_depths']==case['sorted_depths'] for grp in cw.values() for q in grp)
                eq_stable=all(all(invariant(grp,key) for key in strict) for grp in cw.values())
                live_repeat=all(invariant(grp,'issued') for grp in live.values())
                live_changed=live['parent'][0]['issued']!=live['candidate'][0]['issued'];fixed_changed=cw['parent'][0]['issued']!=cw['candidate'][0]['issued']
                agree=all(cw[n][0]['issued']==live[n][0]['issued'] for n in MODES)
                common_enemy=bool(case['opponent_samples']) and all(q['opponent_samples']==case['opponent_samples'] for group in cw.values() for q in group)
                live_no_fallback=all(q['depth'] and q['native_aborted']==0 and q['best'] is not None for grp in live.values() for q in grp)
                matched=bool(parent_restored and exact_work and eq_stable and common_enemy)
                row['check']=dict(live_repeat=live_repeat,live_changed=live_changed,live_no_fallback=bool(live_no_fallback),parent_recovered=parent_restored,equal_work=exact_work,control_repeat=eq_stable,common_opponent_at_fixed_work=common_enemy,agrees_with_live=agree,
                     fixed_changed=fixed_changed,controlled_effect=matched and fixed_changed,robust_live_effect=bool(matched and live_no_fallback and live_repeat and live_changed and fixed_changed and agree),
                     live_candidate_menu_changed=live['parent'][0]['menu']!=live['candidate'][0]['menu'],live_candidate_a0_changed=live['parent'][0]['act0']!=live['candidate'][0]['act0'])
                rows.append(row);print(ticket['seed'],seat,i,row['check'],flush=True)
            tr.step(ob['players'],i,ob['carbon']);tr.set_actions(ob['players'],old);prev=ob['players']
            if i>=max(targets):break
    assert len(rows)==len(prot['tickets'])==16
    totals={k:sum(row['check'][k] for row in rows) for k in rows[0]['check']};clocks={}
    for n in MODES:
        group=[q for row in rows for q in row['live'][n]]
        clocks[n]={}
        for key in ('wall_s','cpu_s','root_policy_wall_s'):
            arr=np.array([q[key] for q in group]);clocks[n][key]=dict(median=float(np.median(arr)),p95=float(np.quantile(arr,.95)),p99=float(np.quantile(arr,.99)),max=float(arr.max()))
        clocks[n].update(calls=len(group),zero_depth=sum(q['depth']==0 for q in group),single_roots=sum(q['single'] for q in group),native_aborted=sum(q['native_aborted'] for q in group),over_058=sum(q['wall_s']>.58 for q in group),over_1=sum(q['wall_s']>1 for q in group))
    result=dict(schema='two-angle/held-public-search/1',protocol_sha256=LOCK,parent_sha256=PARENT,candidate_sha256=CANDIDATE,clock=.58,core=core,live_calls=64,equal_work_calls=64,clock_init_calls=16,opening_cost=opening,
                equal_work_control_sha256=sha(CONTROL.read_bytes()),probe_sha256=sha(Path(__file__).read_bytes()),live_clock_seconds=clocks,counts=totals,roots=rows)
    save(HERE/'roots.json.gz',result);print('COMPLETE',json.dumps(totals),json.dumps(clocks),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu',type=int,required=True);experiment(p.parse_args().cpu)
