"""Replay the observed prefix and invoke each independent public policy from step 286 to terminal."""
from collections import Counter
import copy
import gzip
import json
import os
from pathlib import Path
import sys

from register import AA, HERE, FAMILY, ROOT, RAW, FREIGHT, V4, ORIGINAL, IDS, GUARD, COMMANDS, LADDER, encode, register, require, save, sha
import importlib.util


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def branch(aa,family,source,state,ticket,arm,plan):
    cfg,env,sim0,mods,history,envstate,snapshots,root_calls=state
    sim=copy.deepcopy(sim0);bank=family.official_state.restore(env,envstate)
    for mod,(adapter,memory) in zip(mods,snapshots):
        adapter.restore(mod,memory)
    rows=[];cash=[Counter(),Counter()]
    workers=[(0,IDS[0]),(1,IDS[1]),(1,GUARD)]
    while not env.done:
        obs=[family.public(env,s,bank) for s in (0,1)]
        step=obs[0]['step'];require(plan['step']<=step<299,'Bad suffix length')
        p=obs[0]['players']
        before={u:p[s][2].get(u) for s,u in workers}
        calls=root_calls if step==plan['step'] else [family.measured(m,obs[s],cfg) for s,m in enumerate(mods)]
        native=[dict(a) for a,_ in calls];clocks=[receipt for _,receipt in calls]
        orders=copy.deepcopy(native);cmd=None
        if arm=='intercept_return' and before[IDS[0]] is not None:
            i=step-plan['step']
            proposal=COMMANDS[i] if i<len(COMMANDS) else source.hurry(p,IDS[0],native[0])
            if proposal is not None and source.feasible(p,native[0],IDS[0],proposal):
                cmd=proposal
                if cmd=='NONE':orders[0].pop(IDS[0],None)
                else:orders[0][IDS[0]]=cmd
                mods[0]._ST['tr'].set_actions(p,orders[0])
        require(orders[1]==native[1] and {k for k in set(native[0])|set(orders[0]) if native[0].get(k)!=orders[0].get(k)}<={IDS[0]},'Extraneous changes')
        bank=[b-max(0,c['wall']-1) for b,c in zip(bank,clocks)]
        aa.advance(env,sim,orders)
        flow=family.econ.cash_flow(sim.cash.events)
        for s in (0,1):cash[s].update(flow[s])
        after=sim.obs(0)['players']
        rows.append(dict(step=step,public_sha256=[family.digest(o) for o in obs],commands=native,issued=orders,override=cmd,
                         pre_workers=before,post_workers={u:after[s][2].get(u) for s,u in workers},
                         collisions=sim.last_collisions[:],cash=list(sim.cash),cash_events=sim.cash.events[:],clocks=clocks))
    require(len(rows)<=13 and all(r.status=='DONE' for r in env.states),'Unofficial terminal')
    for s in (0,1):
        family.econ.check_equal(sum(cash[s].values()),sim.cash[s]-history['cash'][s],'suffix wallet')
    score=[r.reward for r in env.states]
    result=dict(stream=ticket['stream'],arm=arm,scores=score,margin=round(score[0]-score[1],2),cash=list(sim.cash),moves=len(rows),
                actual_changes=sum(r['commands'][0]!=r['issued'][0] for r in rows),
                workers={u:dict(terminal=rows[-1]['post_workers'][u],
                                 deposits=[(r['step'],e[2]) for r in rows for e in r['cash_events'] if e[0]=='own_delivery' and e[1]==s and e[3]==u and e[2]>0],
                                 losses=[(r['step'],event) for r in rows for event in r['collisions'] if event[0]==u],
                                 initial_commands=[r['issued'][s].get(u,'NONE') if r['pre_workers'][u] else None for r in rows[:4]]) for s,u in workers},
                clocks=[dict(max_wall=max(r['clocks'][s]['wall'] for r in rows),slow=sum(r['clocks'][s]['wall']>1 for r in rows),
                             total_cpu=sum(r['clocks'][s]['cpu'] for r in rows),total_wall=sum(r['clocks'][s]['wall'] for r in rows),rolls=sum(r['clocks'][s]['rolls'] for r in rows)) for s in (0,1)])
    name=f"stream{ticket['stream']}_{arm}.json.gz"
    digest=save(HERE/name,dict(protocol_sha256=sha((HERE/'protocol.json').read_bytes()),root=history,ticket=ticket,result=result,rows=rows))
    return dict(**result,artifact=name,sha256=digest)


if __name__=='__main__':
    os.sched_setaffinity(0,{4})
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(ROOT/'code/env_src/carbon_challenge-master'))
    aa=load('native_exposed_root',AA/'run.py')
    family=load('shared_native_functions',FREIGHT)
    sys.path.insert(0,str(LADDER/'v4_alphacarbon_decisions'))
    source=load('hurry_public',LADDER/'v4_alphacarbon_decisions/branches.py')
    plan=json.loads((HERE/'protocol.json').read_bytes());require(encode(plan)==encode(register()),'Data changed after registration')
    aa.PLAN=plan;game=json.loads(gzip.decompress(RAW.read_bytes()));results=[]
    for ticket in plan['streams']:
        state=aa.prefix(family,game,ticket['seed_by_seat'],plan['step'])
        for arm in ticket['order']:
            r=branch(aa,family,source,state,ticket,arm,plan)
            results.append(r);print(ticket['stream'],arm,r['scores'],flush=True)
    require(sum(r['moves'] for r in results)<=plan['maximum_joint_transitions'],'Budget')
    save(HERE/'results.json',dict(protocol_sha256=sha((HERE/'protocol.json').read_bytes()),results=results,cpu=sorted(os.sched_getaffinity(0)),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip()))
