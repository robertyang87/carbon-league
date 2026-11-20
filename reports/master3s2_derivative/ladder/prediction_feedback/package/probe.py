"""Sixteen archived same-input roots, real public full search with paired private RNG and natural deadline."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
M3=HERE.parents[2]
BASE=M3/'opponent_policy'
sys.path[:0]=[str(BASE/'search_probe'),str(BASE),str(HERE.parent)]
os.sched_setaffinity(0,{0})
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CARBON_DEBUG']='1'
import numpy as np
import torch
from original_policy import PROTOCOL, module, free_entry
from full_search_check import root_call, assess
import importlib.util
runtime_spec=importlib.util.spec_from_file_location("frozen_feedback_runtime",HERE.parent/"runtime.py")
feedback=importlib.util.module_from_spec(runtime_spec)
runtime_spec.loader.exec_module(feedback)
adjust=feedback.adjust
from build import build, sha


def check_offsets(m,players,op,theta):
    # Score round-trip uses a nonconstant diagnostic tensor at the exact historical occupied cells.
    z=np.arange(225*5,dtype=np.float32).reshape(1,225,5)*np.float32(.017)
    out=m._feedback_adjust(z.copy(),players,op,m._FEEDBACK_WEIGHTS,m._OPP_STEPS,m.FEAT_DIST)
    gold=adjust(z.copy(),players,op,theta,m._OPP_STEPS,m.FEAT_DIST)
    unchanged=m._feedback_adjust(z.copy(),players,op,np.zeros(2,np.float32),m._OPP_STEPS,m.FEAT_DIST)
    assert np.array_equal(unchanged,z) and np.array_equal(out,gold)
    occupied={v[0] for v in players[op][2].values() if v[2]=='COLLECTOR'}
    affected=set(np.flatnonzero(np.any(z[0]!=out[0],axis=1)))
    assert affected==occupied
    return len(affected)


def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    package,info=build()
    assert package==(HERE/'candidate.py').read_bytes()
    assert info==json.loads((HERE/'manifest.json').read_bytes())
    p=(M3/'cold_start/candidate.py').read_bytes()
    m=module(p);n=module(package)
    torch.set_num_threads(1)
    assert m._S_ABORT==n._S_ABORT==.58 and m._TEMP==n._TEMP==1 and m._SEARCH_OK and n._SEARCH_OK
    costs=list(m._S_COST)
    assert all(np.isfinite(v) and v>0 for v in costs)
    weights=np.asarray(info["coefficients"],dtype=np.float32)
    val=[c for c in PROTOCOL['games'] if c['split']=='val']
    assert len(val)==8 and len(set(c['terrain'] for c in val))==4
    signed=json.loads((HERE.parent/'weights.json').read_bytes())
    ladder_held=json.loads((HERE.parent/'held_result.json').read_bytes())
    assert not ({q['terrain'] for q in val} & (set(signed['training_terrains'])|{g['terrain'] for g in ladder_held['games']}))
    rows=[]
    for c in val:
        s=c['candidate_seat']
        tr=m.BeliefTracker(m._Bel,s)
        prev=None
        for t,obs,cfg,shown in free_entry.public_history({**c,'expert_seat':s}):
            if t in (85,215):
                assert 'full_carbon' not in obs and obs['player']==s
                token=f"m3s2-opponent-probe-v1/{c['terrain']}/{s}/{t}".encode()
                seed=int.from_bytes(hashlib.sha256(token).digest()[:4],'big')
                nworkers=check_offsets(n,obs['players'],1-s,weights)
                order=(['original_1','candidate_1','candidate_2','original_2'] if len(rows)%2==0 else
                       ['candidate_1','original_1','original_2','candidate_2'])
                public={k:v for k,v in cfg.items() if k!='randomSeed'}
                arms={key:root_call(n if key.startswith('candidate') else m,tr,prev,obs,public,seed,costs) for key in order}
                passed=assess(arms)
                a,b,c1,c2=[arms[k] for k in ('original_1','original_2','candidate_1','candidate_2')]
                # Same parent root A0, same root menu and native workload across both search versions.
                # When the opponent's moves diverge, native pruning may retain different candidate IDs.
                passed['exact_depth_by_candidate']=all(a['depth_by_arm']==v['depth_by_arm'] for v in (b,c1,c2))
                passed['within_each_version_exact_depth_by_candidate']=(a['depth_by_arm']==b['depth_by_arm'] and c1['depth_by_arm']==c2['depth_by_arm'])
                passed['reproducible_change']=passed['attributable_changed'] and passed['within_each_version_exact_depth_by_candidate']
                rows.append(dict(terrain=c['terrain'],seed=c['seed'],seat=s,step=t,source=c['sha256'],policy_seed=seed,
                                 zero_and_runtime_equivalent=True,enemy_collector_rows=nworkers,order=order,arms=arms,assessment=passed))
            tr.step(obs['players'],t,obs['carbon'])
            tr.set_actions(obs['players'],shown);prev=obs['players']
        print(c['seed'],s,len(rows),flush=True)
    assert len(rows)==16
    counts={k:sum(r['assessment'][k] for r in rows) for k in rows[0]['assessment']}
    return dict(candidate_sha256=info['candidate_sha256'],parent_sha256=sha(p),status='development_mechanism_only',
                costs=costs,history_protocol_sha256=sha((BASE/'protocol.json').read_bytes()),
                probe_sha256=sha(Path(__file__).read_bytes()),start_conditions='Single pin CPU 0; native search is wall-adaptive. All four root calls use identical fixed measured parent warmup costs, public history, same root RNG and temperature 1.',
                times=[85,215],map_count=4,root_count=16,counts=counts,
                all_public_wall_max_s=max(z['wall_s'] for r in rows for z in r['arms'].values()),
                worst_candidate_wall_s=max(z['wall_s'] for r in rows for k,z in r['arms'].items() if k.startswith('candidate')),
                qualified_for_considering_original_development=counts['reproducible_change']>0,
                roots=rows)


if __name__=='__main__':
    out=run()
    raw=json.dumps(out,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    dest=HERE/'roots.json.gz'
    with dest.open('xb') as f:f.write(gzip.compress(raw,mtime=0))
    print(out['counts'],flush=True)
