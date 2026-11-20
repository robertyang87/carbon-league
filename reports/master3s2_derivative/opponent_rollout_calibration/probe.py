"""Full original public entry on unchanged, previously used training prefixes."""
import copy
import importlib.util
import json
import time

import numpy as np

from study import HERE,ROOT,PROTOCOL,initialize,read,put,sha

STEPS=(42,177)
ORDERS=(('parent','neutral','candidate','parent'),('candidate','parent','neutral','candidate'))


def main():
    p=initialize()
    spec=importlib.util.spec_from_file_location('exact_rollout_candidate',HERE/'candidate.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    c._torch.set_num_threads(1);c._net()
    assert c._SEARCH_OK and p._S_ABORT==c._S_ABORT==.58
    beta=dict(c._ROLLOUT_OPP_BETA)
    schema=json.loads((ROOT/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json').read_bytes())
    cfg={k:v['default'] for k,v in schema['configuration'].items() if isinstance(v,dict) and 'default' in v}
    cards=[t for t in json.loads((HERE/'protocol.json').read_bytes())['tickets'] if t['split']=='train']
    pre=list(p._S_COST)
    out=[]
    for card in cards:
        tape,_=read(card);seat=card['seat'];op=1-seat
        tr=p.BeliefTracker(p._Bel,seat);previous=None
        for t,(obs,actual) in enumerate(tape):
            if t in STEPS:
                row=dict(seed=card['seed'],seat=seat,t=t,map=card['map'],runs=[])
                for mode in ORDERS[len(out)%2]:
                    m=p if mode=='parent' else c
                    if mode!='parent':
                        c._ROLLOUT_OPP_BETA=dict.fromkeys(beta,1.) if mode=='neutral' else beta
                    m._ST=dict(tr=copy.deepcopy(tr),prev=previous,rng=np.random.RandomState(card['seed']%1000000+seat*1000+t))
                    m._S_COST[:]=pre;m._S_STATS.pop('last',None)
                    first_opp=[];first_our=[];rolls=[];turns=m._turn;sample=m._sample
                    def turn(o,config,acts):
                        if o['step']==t:
                            first_opp.append(dict(acts[op]));first_our.append(dict(acts[seat]))
                        return turns(o,config,acts)
                    def seen(players,side,lw,lb,rng,step,fixed=None):
                        action=sample(players,side,lw,lb,rng,step,fixed)
                        if step==t+1 and len(rolls)<32:
                            rosters=tuple((k,v[0],v[2]) for k,v in players[side][2].items())
                            rolls.append((side,rosters,dict(action)))
                        return action
                    m._turn=turn;m._sample=seen
                    tic=time.perf_counter()
                    try: issued=m.agent(obs,cfg)
                    finally:m._turn=turns;m._sample=sample
                    elapsed=time.perf_counter()-tic
                    clock=m._S_STATS.get('last')
                    signature=(int(clock[0]),int(clock[1]),int(clock[2])) if clock else (1,0,0)
                    row['runs'].append(dict(mode=mode,elapsed=elapsed,work=signature,order=issued,
                        root_enemy=first_opp,root_own=first_our,first_continuations=rolls))
                pa=next(r for r in row['runs'] if r['mode']=='parent')
                zero=next(r for r in row['runs'] if r['mode']=='neutral')
                new=next(r for r in row['runs'] if r['mode']=='candidate')
                allruns=row['runs']
                same=all(r['work']==pa['work'] for r in allruns)
                repeat=allruns[0]['order']==allruns[-1]['order']
                row['all_work_equal']=same;row['first_arm_repeat_identical']=repeat
                row['root_enemy_orders_equal']=all(r['root_enemy']==pa['root_enemy'] for r in allruns)
                row['root_own_menus_equal']=all(r['root_own']==pa['root_own'] for r in allruns)
                row['neutral_issued_identical']=pa['order']==zero['order']
                row['matched_issued_change']=same and repeat and row['root_enemy_orders_equal'] and row['root_own_menus_equal'] and row['neutral_issued_identical'] and new['order']!=pa['order']
                # Only compare the same initial branch and physical role before either continuation changes state.
                before=pa['first_continuations'];after=new['first_continuations']
                shared=[(a,b) for a,b in zip(before,after) if a[0]==b[0] and a[1]==b[1]] if row['root_own_menus_equal'] and row['root_enemy_orders_equal'] else []
                row['first_continuations_on_same_branch']={str(s):dict(compared=sum(a[0]==s for a,b in shared),
                    changed=sum(a[0]==s and a[2]!=b[2] for a,b in shared)) for s in (seat,op)}
                out.append(row)
            tr.step(obs['players'],t,obs['carbon']);tr.set_actions(obs['players'],actual[seat]);previous=obs['players']
    for r in out:print(r['seed'],r['seat'],r['t'],r['all_work_equal'],r['neutral_issued_identical'],r['matched_issued_change'],r['first_continuations_on_same_branch'])
    meta=dict(steps=STEPS,protocol=PROTOCOL,source_parent=sha((HERE.parent/'cold_start/candidate.py').read_bytes()),
          source_candidate=sha((HERE/'candidate.py').read_bytes()),rows=out,
          all_four_work_matches=sum(r['all_work_equal'] for r in out),repeat_matches=sum(r['first_arm_repeat_identical'] for r in out),
          public_issued_changes_at_matched_work=sum(r['matched_issued_change'] for r in out),initial_costs=pre)
    print(put(HERE/'probe.json',meta))
if __name__=='__main__':main()
