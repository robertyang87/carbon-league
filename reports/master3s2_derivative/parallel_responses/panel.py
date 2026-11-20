"""Run native M1 and M2 from matched public histories, retaining real adaptive budgets and sampled scores."""
import copy
import gzip
import importlib.util
import json
import os
import sys
import time
import zipfile
import numpy as np
from register import HERE,FAMILY,PARENT,sha

LOCK='8198ba709f101758116f1e83224080e03278124b010bb73abbc0b6cce01a29f9'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def put(path,data):
    raw=(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    tmp=path.with_suffix('.partial');tmp.write_bytes(raw);os.replace(tmp,path)

def sources(card):
    path=FAMILY/card['source'];assert sha(path.read_bytes())==card['sha256']
    with zipfile.ZipFile(path) as z:
        raw=z.read('parent/receipt.json');assert sha(raw)==card['registration_receipt_sha']
        receipt=json.loads(raw);reg=z.read('parent/registration.json')
        assert sha(reg)==receipt['protocol_sha256']
        reg=json.loads(reg);assert reg['physical_map_sha256']==card['terrain'] and reg['files']['opponent']['sha256']==PLAN['examiner']
        raw=z.read('parent/turns.jsonl');assert sha(raw)==receipt['turns_sha256']
        turns=list(map(json.loads,raw.splitlines()))
        frames=list(map(json.loads,gzip.decompress(z.read('parent/observations.jsonl.gz')).splitlines()))
    assert len(turns)==len(frames)==299
    return reg,turns,frames

def cfg():
    schema=json.loads((FAMILY.parents[1]/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json').read_bytes())['configuration']
    return {k:v['default'] for k,v in schema.items() if isinstance(v,dict) and 'default' in v}|{'episodeSteps':300}

def capture(m,tracker,previous,obs,config,rng,cost,M):
    m._ST=dict(tr=copy.deepcopy(tracker),prev=previous,rng=np.random.RandomState(rng))
    m._S_M=m._S_MMAX=M
    m._S_COST[:]=cost
    frame={}
    def hook(_,offset,returned):
        frame.update(sys._getframe(1).f_locals)
    mon=sys.monitoring;slot=4
    mon.use_tool_id(slot,'native final return')
    try:
        mon.register_callback(slot,mon.events.PY_RETURN,hook);mon.set_local_events(slot,m._search.__code__,mon.events.PY_RETURN)
        w=time.perf_counter();p=time.process_time();a=m.agent(obs,config);cpu=time.process_time()-p;wall=time.perf_counter()-w
    finally:
        mon.set_local_events(slot,m._search.__code__,0);mon.register_callback(slot,mon.events.PY_RETURN,None);mon.free_tool_id(slot)
    assert m._S_ABORT==.58 and m._TEMP==1.
    cands=frame.get('cands',[a]);alive=frame.get('alive',[0]);depth=frame.get('depth',0)
    dep=frame.get('dep',{});vals=frame.get('val',{});op=frame.get('opp',{})
    dv={c:[float(vals[c,i]) for i in range(M)] for c in alive if all((c,i) in vals for i in range(M))}
    def winner(values):
        first=max(alive,key=values.__getitem__)
        return 0 if first==0 or values[first]-values[0]<=m._S_EPS else first
    best=cands.index(a);diag=None
    if len(dv)==len(alive) and depth>0:
        assert best==winner({c:float(np.mean(v)) for c,v in dv.items()})
        diag=[winner({c:v[i] for c,v in dv.items()}) for i in range(M)]
    return dict(issued=a,chosen=best,menu=cands,alive=alive,depth_per=dep,values=dv,individual_winners=diag,
                sample_enemy=op,aborted=depth==0,cut=frame.get('cut',True),cost_after=list(m._S_COST),
                root_cap=frame.get('most',1),clock=dict(wall=wall,cpu=cpu,H=depth,K=len(cands),rolls=frame.get('rolls',0)))

def compare(card,ordinal,m,config):
    _,turns,frames=sources(card);side,seed=card['seat'],card['seed'];tracker=m.BeliefTracker(m._Bel,side)
    prev=None;after_warm={};warm_seed=(seed+side*151)%2**32
    rows=[];steps=PLAN['steps'];warm_steps={t-1 for t in steps}
    for i,(line,fr) in enumerate(zip(turns,frames,strict=True)):
        obs=fr['observations'][side]
        assert obs['step']==i and 'full_carbon' not in obs
        assert sha((json.dumps(obs,separators=(',',':'))+'\n').encode())==fr['public_observation_sha256'][side]==line['public_observation_sha256'][side]
        if i in warm_steps:
            warm=capture(m,tracker,prev,obs,config,warm_seed+i,[.001,.002,.001,.002],1)
            after_warm[i+1]={'cost':warm['cost_after'],'cost_seed':warm_seed+i,'prior_clock':warm['clock']}
        if i in steps:
            idx=ordinal*len(steps)+steps.index(i);rseed=(seed+19073+i*7+side*17)%2**32
            order=[1,2] if idx%2==0 else [2,1]
            rec=dict(step=i,public_sha=fr['public_observation_sha256'][side],pair_order=order,policy_seed=rseed,**after_warm[i])
            for n in order:
                rec[f'M{n}']=capture(m,tracker,prev,obs,config,rseed,rec['cost'],n)
            one,two=rec['M1'],rec['M2']
            assert one['menu'][0]==two['menu'][0]
            if one['sample_enemy'] and two['sample_enemy']: assert one['sample_enemy'][0]==two['sample_enemy'][0]
            # Reveal actual opponent action to the evaluator only after both native outputs.
            enemy=line['calls'][1-side]['response']['action']
            pp=obs['players'][1-side]
            agents=list(pp[1])+list(pp[2])
            rec['real_examiner_root']=enemy;rec['enemy_entities']=agents
            rec['actual_enemy_search_last']=line['calls'][1-side]['response'].get('search_last')
            rows.append(rec)
        orders=line['calls'][side]['response']['action']
        tracker.step(obs['players'],i,obs['carbon']);tracker.set_actions(obs['players'],orders)
        prev=obs['players']
    put(HERE/'roots'/f'{seed}_{side}.json.gz',dict(**card,protocol=LOCK,rows=rows))
    print(seed,side,[(r['step'],r['M1']['clock']['K'],r['M2']['clock']['K'],r['M1']['issued']!=r['M2']['issued']) for r in rows],flush=True)

raw_plan=(HERE/'protocol.json').read_bytes();assert sha(raw_plan)==LOCK
PLAN=json.loads(raw_plan)
if __name__=='__main__':
    assert os.sched_getaffinity(0)=={4} and sha(PARENT.read_bytes())==PLAN['parent']
    m=load('parallel_native_v2',PARENT)
    assert m._SEARCH_OK and m._S_M==m._S_MMAX==1 and m._RX
    m._torch.set_num_threads(1)
    config=cfg()
    for i,card in enumerate(PLAN['tickets']):
        p=HERE/'roots'/f"{card['seed']}_{card['seat']}.json.gz"
        if not p.exists():compare(card,i,m,config)
