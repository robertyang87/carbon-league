"""Natural 0.58-second search on both seats of held maps; distinguish time scatter from changed issued orders."""
import copy
import hashlib
import json
import time
import numpy as np
from build import assemble
from data import HERE,D,P,save,module,read,sha

def replay_one(m,method,obs,cfg,tr,previous,key,init):
    t=obs["step"];s=obs["player"]
    m._ST=dict(tr=copy.deepcopy(tr),prev=previous,rng=np.random.RandomState(key))
    m._S_COST[:]=init;m._S_REC=[];m._S_STATS.pop('last',None);native=m._turn;enemy=[]
    def turn(o,config,acts):
        if o['step']==t:enemy.append(acts[1-s].copy())
        return native(o,config,acts)
    m._turn=turn;started=time.perf_counter()
    try:issued=m.agent(obs,cfg)
    finally:m._turn=native
    delta=time.perf_counter()-started;last=m._S_STATS.get('last')
    assert last and len(m._S_REC)==1
    sr=m._S_REC[0]
    assert all(a==enemy[0] for a in enemy) and sr[0]==t and sr[1]==s
    return dict(mode=method,seconds=delta,issued=issued,K=last[0],H=last[1],M=last[2],
       depths=sr[3],first_menu=sr[2][0],selected=sr[5],menu=sr[2],enemy=enemy[0])

def main():
    arms={k:module(assemble(k),'replay_'+k) for k in ('zero','bc','awr')}
    par=arms['zero'];init=list(par._S_COST)
    assert all(m._TEMP==1. and m._S_ABORT==.58 and m._SEARCH_OK for m in arms.values())
    schema=json.loads((D.parents[1]/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json').read_bytes())
    cfg={k:v['default'] for k,v in schema['configuration'].items() if isinstance(v,dict) and 'default' in v}
    out=[]
    for card in [c for c in P['tickets'] if c['split']=='held']:
        tape,_=read(card);s=card['seat']
        tr=par.BeliefTracker(par._Bel,s);previous=None
        for t,(obs,active) in enumerate(tape):
            if t in (46,165):
                row=dict(map=card['map'],seed=card['seed'],t=t,seat=s,real=active[s],runs=[])
                key=int.from_bytes(hashlib.sha256(f"awr-native/{card['archive_sha256']}/{s}/{t}".encode()).digest()[:4],'big')
                for method in (('zero','bc','awr','zero') if len(out)%2==0 else ('zero','awr','bc','zero')):
                    row['runs'].append(replay_one(arms[method],method,obs,cfg,tr,previous,key,init))
                z,b,a=[next(r for r in row['runs'] if r['mode']==k) for k in ('zero','bc','awr')]
                replay=row['runs'][-1]
                row.update(control_stable=(z['issued']==replay['issued'] and z['menu']==replay['menu']),
                     same_work=all((u['K'],u['H'],u['M'],u['depths'])==(z['K'],z['H'],z['M'],z['depths']) for u in row['runs']),
                     same_enemy=all(u['enemy']==z['enemy'] for u in row['runs']),
                     changed_menu=a['menu']!=z['menu'],changed_first=a['first_menu']!=z['first_menu'],
                     changed_vs_parent=a['issued']!=z['issued'],changed_vs_bc=a['issued']!=b['issued'])
                row['attributed_vs_parent']=row['control_stable'] and row['same_work'] and row['same_enemy'] and row['changed_vs_parent']
                row['attributed_unique_vs_both']=row['attributed_vs_parent'] and row['changed_vs_bc']
                out.append(row)
            tr.step(obs['players'],t,obs['carbon']);tr.set_actions(obs['players'],active[s]);previous=obs['players']
    counts={k:sum(r[k] for r in out) for k in ['control_stable','same_work','same_enemy','changed_menu','changed_first','changed_vs_parent','changed_vs_bc','attributed_vs_parent','attributed_unique_vs_both']}
    maps=sorted(set(r['map'] for r in out if r['attributed_vs_parent']))
    print(counts,len(maps),flush=True)
    save(HERE/'native.json.gz',dict(candidate=sha(assemble('awr')),baseline=P['parent'],counts=counts,qualified_maps=maps,source_roots=out,initial_costs=init))

if __name__=='__main__':main()
