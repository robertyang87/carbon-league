"""Select only by past-observable sampled rival-order disagreement; run a same-root hardware-timed crossover."""
import copy
import gzip
import json
import os
import sys
from pathlib import Path
import hashlib

os.sched_setaffinity(0,{0})
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CARBON_DEBUG']='1'
import numpy as np
import torch
from probe import HERE,M3,module,root_call,assess,build,sha


def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    locked=(HERE/'conditional_protocol.json').read_bytes()
    assert sha(locked)=='3405477c48857075b07d6ec041131f94dcc5bc23efa720be63dbb9833055997a'
    plan=json.loads(locked)
    code,info=build()
    assert sha(code)==plan['candidate_sha256'] and info['parent_sha256']==plan['parent_sha256']
    m=module((M3/'cold_start/candidate.py').read_bytes())
    n=module(code)
    torch.set_num_threads(1)
    old=json.loads(gzip.decompress((HERE/'roots.json.gz').read_bytes()))
    assert old['candidate_sha256']==sha(code)
    costs=old['costs']
    used=set(json.loads((HERE.parent/'weights.json').read_bytes())['training_terrains'])
    used.update(e['terrain'] for e in json.loads((HERE.parent/'held_result.json').read_bytes())['games'])
    summaries=[]; tickets=[]
    for g in plan['games']:
        data=(M3/'ladder/replays'/(g['match']+'-1.json.gz')).read_bytes()
        assert sha(data)==g['replay_sha256']
        tape=json.loads(gzip.decompress(data))
        exposed=json.loads((M3/'ladder/exposed_ladder_maps.json').read_bytes())
        ticket=next(r for r in exposed if r['match_id']==g['match'])
        physical=sha(bytes(map(int,tape['steps'][0][0]['observation']['full_carbon'])))
        assert ticket['scorecard_map_sha256']==g['terrain'] and ticket['replay_gzip_sha256']==sha(data)
        assert physical==ticket['opening_carbon_bytes_sha256'] and physical not in used
        used.add(physical)
        frames=tape['steps'];s=g['seat'];enemy=1-s
        assert len(frames)==300
        tr=m.BeliefTracker(m._Bel,s);previous=None
        windows=[];changed_team=0;with_collector=0
        for t,frame in enumerate(frames[:-1]):
            v=frame[s]['observation'];both=frame[0]['observation']
            assert both['step']==t and v['player']==s
            obs=dict(step=t,player=s,players=both['players'],trees=both['trees'],carbon=v['carbon'],remainingOverageTime=v['remainingOverageTime'])
            roster=obs['players'];collector=[k for k,u in roster[enemy][2].items() if u[2]=='COLLECTOR']
            if collector:with_collector+=1
            prior=copy.deepcopy(tr) if len(windows)<4 else None
            est,_=tr.step(roster,t,obs['carbon'])
            if collector:
                x=m._quant(m.encode(roster,t,enemy,est,[1.]*m.N,previous))[None]
                lw,lb,_=m._net()(x.astype(np.float32),m._SID)
                lw,lb=m._opponent_root_readout(x,lw,lb,roster,enemy)
                adjusted=n._feedback_adjust(lw.copy(),roster,enemy,n._FEEDBACK_WEIGHTS,n._OPP_STEPS,n.FEAT_DIST)
                seed=int(1000003*(t+1)+17*s)%(2**31)+7
                forecast=[m._sample(roster,enemy,z[0],lb[0],np.random.RandomState(seed),t) for z in (lw,adjusted)]
                if forecast[0]!=forecast[1]:
                    changed_team+=1
                    if any(forecast[0].get(uid)!=forecast[1].get(uid) for uid in collector):
                        if len(windows)<4:
                            tickets.append((dict(match=g['match'],opponent=g['opponent'],terrain=physical,scorecard_map_sha256=g['terrain'],seat=s,step=t),
                                            prior,previous,obs,{k:a for k,a in tape['configuration'].items() if k!='randomSeed'},forecast))
                        windows.append(t)
            tr.set_actions(roster,frames[t+1][s]['action'] or {})
            previous=roster
        summaries.append(dict(**g,opening_carbon_bytes_sha256=physical,all_decisions=len(frames)-1,with_enemy_collector=with_collector,
                              rival_team_changed=changed_team,candidate_collection_changes=len(windows),
                              all_conditional_steps=windows,selected_steps=windows[:4]))
    rows=[]
    for idx,(r,tr,prev,obs,cfg,forecast) in enumerate(tickets):
        seed=int.from_bytes(hashlib.sha256(f"{r['match']}/{r['step']}".encode()).digest()[:4],'big')
        tags=(['original_1','candidate_1','candidate_2','original_2'] if idx%2==0 else ['candidate_1','original_1','original_2','candidate_2'])
        arms={tag:root_call(n if tag.startswith('candidate') else m,tr,prev,obs,cfg,seed,costs) for tag in tags}
        check=assess(arms)
        a,b,x,y=[arms[t] for t in ('original_1','original_2','candidate_1','candidate_2')]
        repeated=(a['depth_by_arm']==b['depth_by_arm'] and x['depth_by_arm']==y['depth_by_arm'])
        samples=all(arms[tag]['opponent_samples']==[forecast[int(tag.startswith('candidate'))]] for tag in tags)
        check.update(reproducible_change=bool(check['attributable_changed'] and repeated and samples),
                     all_actual_search_samples_equal_screen=samples,within_version_repeat_depth_by_id=repeated,
                     all_four_exact_depth_by_candidate=all(z['depth_by_arm']==a['depth_by_arm'] for z in [b,x,y]))
        rows.append(dict(**r,private_rng=seed,order=tags,screening_forecasts=forecast,arms=arms,assessment=check))
        print(r['opponent'],r['step'],check['reproducible_change'],flush=True)
    counts={k:sum(r['assessment'][k] for r in rows) for k in rows[0]['assessment']} if rows else {}
    out=dict(protocol_sha256=sha(locked),probe_sha256=sha(Path(__file__).read_bytes()),candidate_sha256=sha(code),
             parent_sha256=info['parent_sha256'],parent_measured_costs=costs,tickets=summaries,counts=counts,roots=rows,
             max_candidate_wall_s=max((v['wall_s'] for r in rows for tag,v in r['arms'].items() if tag.startswith('candidate')),default=None))
    path=HERE/'conditional_roots.json.gz'
    with path.open('xb') as f:f.write(gzip.compress(json.dumps(out,sort_keys=True,allow_nan=False,separators=(',',':')).encode(),mtime=0))
    print(counts,flush=True)


if __name__=='__main__':run()
