"""Check disabled-parent identity at fixed work, and all changed destinations on the archived joint orders."""
import copy
import gzip
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import time
from probe import HERE,D,PARENT,STEPS,RUN,build,module,torch,np,trial,sha


def occupied(mod,players,s,actions):
    cash,bases,workers,trees=players[s]
    count=len(workers)
    cells=Counter(mod.move(w[0],mod.WIDX[actions.get(uid) or 'NONE']) for uid,w in workers.items())
    for uid,pos in bases.items():
        if actions.get(uid) in ('RECPLANTER','RECCOLLECTOR') and count<10 and cash>=30:
            cells[pos]+=1; cash-=30; count+=1
    return cells


def run():
    code,manifest=build()
    m,n=module(PARENT.read_bytes()),module(code)
    torch.set_num_threads(1)
    tape=json.loads(gzip.decompress((D/'ladder/replays'/(RUN+'-1.json.gz')).read_bytes()))
    costs=json.loads(gzip.decompress((HERE/'makabaka_roots.json.gz').read_bytes()))['private_cost_snapshot']
    cfg={k:v for k,v in tape['configuration'].items() if k!='randomSeed'}
    tr=m.BeliefTracker(m._Bel,1); prev=None
    samples=[]; changed=[]; danger_sites=0
    for t,(before,after) in enumerate(zip(tape['steps'],tape['steps'][1:])):
        s=1;v=before[s]['observation'];both=before[0]['observation']
        obs=dict(player=s,step=t,players=both['players'],trees=both['trees'],carbon=v['carbon'],remainingOverageTime=v['remainingOverageTime'])
        p=obs['players'];a=after[s]['action'] or {}
        n._CARGO_TRACE={}
        transformed=n._cargo_postsearch(obs,prev,a,time.perf_counter())
        checked=n._CARGO_TRACE.get('checked',[])
        danger_sites+=len(checked)
        if transformed!=a:
            slots=occupied(n,p,s,transformed)
            uids=[u for u in set(transformed)|set(a) if transformed.get(u)!=a.get(u)]
            assert all(u in p[s][2] and slots[n.move(p[s][2][u][0],n.WIDX[transformed.get(u) or 'NONE'])]==1 for u in uids)
            assert all(a.get(b)==transformed.get(b) for b in p[s][1])
            changed.append(dict(step=t,changed_workers=sorted(uids)))
        n._CARGO_TRACE=None
        if t in STEPS:
            for seed in (205,818):
                rows=[]
                for agent,mode in ((m,'P'),(n,'OFF')):
                    wall=agent._time
                    # Identical monotone budget-clock reads remove scheduler variance. Public-entry
                    # behavior here validates disabled control only; see probe.py for real clocks.
                    ticks=[0]
                    def clock():
                        ticks[0]+=1
                        return ticks[0]*.0002
                    agent._time=SimpleNamespace(perf_counter=clock)
                    try:
                        row=trial(agent,tr,prev,obs,cfg,seed,costs,mode)
                        rows.append((row['issued'],row['search'],ticks[0]))
                    finally:
                        agent._time=wall
                assert rows[0]==rows[1]
                samples.append(dict(step=t,private_seed=seed,clock_queries=rows[0][2],
                                    physical_order=rows[0][0],completed_depth=rows[0][1]['completed_depth'],
                                    candidate_count=len(rows[0][1]['menu'])))
        tr.step(p,t,obs['carbon']);tr.set_actions(p,a);prev=p
    return dict(candidate_sha256=manifest['candidate_sha256'],source_parent_sha256=manifest['parent_sha256'],
                checked_steps=299,checked_exposed_worker_opportunities=danger_sites,
                candidate_cells_without_new_friendly_conflicts=True,changed_recorded_joint_orders=changed,
                fixed_work_source_off_exact=samples)


if __name__=='__main__':
    result=run()
    (HERE/'parity.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(len(result['fixed_work_source_off_exact']),len(result['changed_recorded_joint_orders']),result['checked_exposed_worker_opportunities'])
