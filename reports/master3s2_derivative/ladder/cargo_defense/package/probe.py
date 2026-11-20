"""Exercise the unchanged, disabled and enabled public agents on the two named historical turns."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
D=HERE.parents[2]
sys.path.insert(0,str(D/'opponent_policy'))
os.sched_setaffinity(0,{0})
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CARBON_DEBUG']='1'
import numpy as np
import torch
from original_policy import module
from build import build,sha,PARENT
RUN='160624b76d6278dfb1e6ae1d145edce0'
STEPS=(138,140)
FOCAL={138:'player-1-worker-3',140:'player-1-worker-11'}


def stored(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def trial(m,pre,previous,obs,cfg,seed,costs,mode):
    m._ST=dict(tr=copy.deepcopy(pre),prev=previous,rng=np.random.RandomState(seed))
    m._S_COST[:]=costs
    m._S_STATS.pop('last',None)
    m._S_REC=[]
    trace={}
    if mode!='P':
        m._CARGO_DEFENSE=mode=='ON'
        m._CARGO_TRACE=trace
    start=time.perf_counter()
    issued=m.agent(obs,cfg)
    wall=time.perf_counter()-start
    if mode!='P':m._CARGO_TRACE=None
    records=m._S_REC
    if not records:
        return dict(mode=mode,issued=issued,before=trace.get('before',issued),wall_s=wall,search=None,trace=trace)
    assert len(records)==1
    _,_,menu,depth,score,choice=records[0]
    before=menu[choice]
    if mode=='ON':
        assert before==trace['before']
    else:
        assert before==issued and not trace
    stats=m._S_STATS['last']
    return dict(mode=mode,issued=issued,before=before,wall_s=wall,search=dict(menu=menu,
                depths=depth,scores=score,chosen=choice,completed_depth=stats[1],rollouts=stats[2]),trace=trace)


def run():
    code,manifest=build()
    assert code==(HERE/'candidate.py').read_bytes()
    parent=module(PARENT.read_bytes());candidate=module(code)
    torch.set_num_threads(1)
    truth=stored(HERE.parent/'labels.py','cargo_feedback_outcomes')
    native=stored(HERE.parent/'runtime.py','cargo_training_risk')
    frozen=json.loads((HERE.parent/'weights.json').read_bytes());w=np.asarray(frozen['weights'],np.float32)
    raw=(D/'ladder/replays'/(RUN+'-1.json.gz')).read_bytes()
    assert sha(raw)=='d6e327fa8846419a895bba9b0af8469eb188b4cd1a0057be68947c4be23807e2'
    tape=json.loads(gzip.decompress(raw)); cfg={k:v for k,v in tape['configuration'].items() if k!='randomSeed'}
    assert all(cfg[k]==v for k,v in truth.RULES.items())
    seat=1
    tracker=parent.BeliefTracker(parent._Bel,seat);previous=None;rows=[];latencies=[]
    old_cost=json.loads(gzip.decompress((D/'ladder/prediction_feedback/package/roots.json.gz').read_bytes()))
    costs=old_cost['costs']
    frames=tape['steps']
    for t,(frame,after) in enumerate(zip(frames,frames[1:])):
        joint=[p['action'] or {} for p in after]
        actual=joint[seat]
        v=frame[seat]['observation']; common=frame[0]['observation'];players=common['players']
        obs=dict(player=seat,step=t,players=players,trees=common['trees'],carbon=v['carbon'],remainingOverageTime=v['remainingOverageTime'])
        if t in STEPS:
            group=[];uid=FOCAL[t]
            sr=native.prepare(players,seat,t,previous);f=native.forecast(sr,uid,w,frozen['recruit_probability'])
            embedded=candidate._cargo_forecast(candidate._cargo_prepare(players,seat,t,previous),uid,candidate._CARGO_W,candidate._CARGO_RECRUIT)
            assert embedded==f
            # Controlled seeds only for this comparison; live traffic used its own natural RNG and timing.
            for j in (0,1):
                seed=int.from_bytes(hashlib.sha256(f'cargo/risk/{RUN}/{t}/{j}'.encode()).digest()[:4],'big')
                tags=['P','OFF','ON','ON','OFF','P'] if j==0 else ['ON','OFF','P','P','OFF','ON']
                arms=[trial(parent if tag=='P' else candidate,tracker,previous,obs,cfg,seed,costs,tag) for tag in tags]
                controls=[a for a in arms if a['mode']!='ON'];changes=[a for a in arms if a['mode']=='ON'];p=controls[0]
                matched=bool(p['search']) and all(a['search']==p['search'] and a['before']==p['before'] for a in arms)
                eq=all(a['issued']==p['issued'] for a in controls)
                focal_changed=changes[0]['issued'].get(uid)!=p['issued'].get(uid)
                replicated=all(a['issued']==changes[0]['issued'] for a in changes)
                labels=truth.label_actions(players,seat,uid,joint,after[0]['observation']['players'],cfg)
                alternatives=[a['issued'].get(uid) or 'NONE' for a in [p,changes[0]]]
                reasons=[int(labels['labels'][truth.ACTIONS.index(z)]) for z in alternatives]
                group.append(dict(seed=seed,order=tags,matched_complete_search=matched,off_repeats_equal_parent=eq,
                                  on_repeat_stable=replicated,focal_order_comparison=alternatives,
                                  counterfactual_enemy_caused_death_under_historical_other_orders=reasons,
                                  native_equals_historical_complete=p['issued']==actual,
                                  genuine_reproducible_joint_change=bool(matched and eq and replicated and changes[0]['issued']!=p['issued']),
                                  genuine_reproducible_focal_change=bool(matched and eq and replicated and focal_changed),arms=arms))
            candidate._CARGO_TRACE=None
            assert candidate._cargo_postsearch(obs,previous,actual,time.perf_counter()-.60) is actual
            conditioned=candidate._cargo_postsearch(obs,previous,actual,time.perf_counter())
            rows.append(dict(step=t,focal=uid,archive_issued=actual,archive_full_reprocessed=conditioned,
                             focal_risk=f['risk'],guard_focal_output_given_archived_order=conditioned.get(uid) or 'NONE',
                             archive_focal_order=actual.get(uid) or 'NONE',paired_seeds=group))
        # A separate timing census of the guard alone against observed own joint orders. Its input is
        # the own-search-output surrogate; these numbers measure overhead, not actual controlled play.
        candidate._CARGO_TRACE=None
        start=time.perf_counter();candidate._cargo_postsearch(obs,previous,actual,start);latencies.append(time.perf_counter()-start)
        tracker.step(players,t,obs['carbon']);tracker.set_actions(players,actual);previous=players
    assert len(latencies)==299
    return dict(candidate_sha256=manifest['candidate_sha256'],parent_sha256=manifest['parent_sha256'],
                source_gzip_sha256=sha(raw),protocol="Both given exposed roots, two deterministically hashed private RNG states, symmetrical P/OFF/ON/ON/OFF/P ordering reversed on second private state; exact underlying primary search and repeated issued orders for attribution. No retries.",
                private_cost_snapshot=costs,rows=rows,guard_latency_all_299_ms=dict(mean=1000*np.mean(latencies),p95=1000*np.percentile(latencies,95),
                p99=1000*np.percentile(latencies,99),max=1000*max(latencies)),positive_qualifying=sum(g['genuine_reproducible_focal_change'] for r in rows for g in r['paired_seeds']))


if __name__=='__main__':
    result=run()
    with (HERE/'makabaka_roots.json.gz').open('xb') as f:
        f.write(gzip.compress(json.dumps(result,sort_keys=True,allow_nan=False,separators=(',',':')).encode(),mtime=0))
    for r in result['rows']:
        print(r['step'],r['archive_focal_order'],r['guard_focal_output_given_archived_order'],[(g['matched_complete_search'],g['focal_order_comparison'],g['genuine_reproducible_focal_change']) for g in r['paired_seeds']])
    print(result['guard_latency_all_299_ms'])
