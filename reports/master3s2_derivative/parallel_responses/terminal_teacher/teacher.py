"""Evaluate two historically fixed team commands under paired native neural continuations, from own public belief."""
import argparse
import ast
import json
import math
import os
import sys
import time
import numpy as np

from register import HERE,D,OLD,CLOSE,key,read,sha
sys.path.insert(0,str(D/'opponent_policy'))
from original_policy import PROTOCOL,free_entry,module

LOCK='7fc66af456ef7458aae2a53c2c0cdd86db90b4e192931eb540f3d74c5481998f'
PLAN=read(HERE/'protocol.json')
assert sha((HERE/'protocol.json').read_bytes())==LOCK

def must_replace(text,a,b):
    if text.count(a)!=1:raise ValueError('Native source anchor differs: '+a)
    return text.replace(a,b)

def from_native(path):
    raw=path.read_bytes();text=raw.decode().replace('\r\n','\n')
    native=module(raw)
    native.__file__=str(path)
    native._torch.set_num_threads(1)
    tree=ast.parse(text);node,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_search']
    src='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    a=src.index('    most = _S_K');b=src.index('    # the opponent',a)
    src=src[:a]+'    cands = _fixed_orders\n'+src[b:]
    src=must_replace(src,'M = _S_M; D2 = _S_D2','M = len(_draw_seeds); D2 = 8')
    src=must_replace(src,'seed0 + 7 + 31 * m','_draw_seeds[m] + 7')
    src=must_replace(src,'seed0 + 7919 * t + 31 * m + 1','_draw_seeds[m] + 7919 * t + 1')
    src=must_replace(src,'seed0 + 7919 * t + 31 * m + 2','_draw_seeds[m] + 7919 * t + 2')
    a=src.index('    def late(rows, steps):');b=src.index('    alive = ',a)
    src=src[:a]+'''    def late(rows, steps):
        return False

    hire = min(configuration.get('recCollectorCost',30),configuration.get('recPlanterCost',30))
    def eliminated(p):
        return not p[2] and not p[3] and (not p[1] or p[0]<hire)

    def over(b):
        return b[2]['step']>LAST_TURN or any(eliminated(p) for p in b[2]['players'])

    def result(b):
        state=b[2]
        flags=[eliminated(p) for p in state['players']]
        if any(flags):
            scores=[state['step']-301 if gone else b[3][i][0] for i,gone in enumerate(flags)]
        else:
            scores=[p[0] for p in state['players']]
        return _terminal(scores[me]-scores[op])

'''+src[b:]
    end=src.index('    while not cut and depth == D2')
    src=src[:end]+'''    _result['values']=[[float(val[c,m]) for c in (0,1)] for m in range(M)]
    _result['opponent_root']=[opp[m] for m in range(M)]
    _result['final_steps']=[[b[2]['step'] for b in boards if b[1]==m] for m in range(M)]
    _result['exact']=[[over(b) for b in boards if b[1]==m] for m in range(M)]
    return cands[0]
'''
    local={};exec(compile(src,'research fixed pair of '+path.name,'exec'),native.__dict__,local)
    native._fixed_pair=local['_search'];native._fixed_source=sha(src.encode())
    return native

def restore_old(card, model):
    book=next(g for g in PROTOCOL['games'] if g['sha256']==card['source_sha'])
    root=read(D/card['root_path'])
    tr=model.BeliefTracker(model._Bel,card['seat']);prev=None
    for t,obs,cfg,issued in free_entry.public_history(book):
        if t==card['step']:
            saved=root['public_root']
            assert all(obs[k]==saved[k] for k in ('players','step','player','trees','carbon'))
            est,_=tr.step(obs['players'],t,obs['carbon'])
            # Native simultaneous sampling assigns RNG draws in the observation's original unit order.
            # The archival JSON for the root dictionary sorts its keys. Use the recovered public order.
            obs=dict(obs,remainingOverageTime=saved['remainingOverageTime'])
            assert sha((json.dumps(obs,separators=(',',':'))+'\n').encode())==root['authority']['public_sha256_by_seat'][card['seat']]
            return obs,cfg,est,prev,[a['order'] for a in root['arms']],[a['original_value'] for a in root['arms']]
        tr.step(obs['players'],t,obs['carbon']);tr.set_actions(obs['players'],issued);prev=obs['players']
    raise ValueError('Source does not reach registered step')

def simulate(model,observed,cfg,est,prev,commands,seeds):
    model._fixed_orders=commands
    model._draw_seeds=seeds
    model._result={}
    start=time.perf_counter();cpu=time.process_time()
    model._fixed_pair(observed,cfg,est,commands[0],None,None,prev,start)
    result=model._result
    result.update(wall=time.perf_counter()-start,cpu=time.process_time()-cpu)
    assert np.shape(result['values'])==(len(seeds),2) and np.isfinite(result['values']).all()
    return result


def evaluate(card,mod):
    row=dict(**card,protocol=LOCK,simulator_source_sha=mod._fixed_source)
    root_path=D/card['root_path'];assert sha(root_path.read_bytes())==card['root_sha']
    if card['status']=='skipped':
        return row|dict(status='historical_skip',reason=read(root_path)['reason'])
    if card['group']=='close':
        from restore_close import restore
        ob,cfg,est,prev,cmd,original=restore(card,mod)
        menu=read(root_path)['rows'][card['close_ordinal']]
        assert menu['ticket']['seed']==card['seed']
        if np.isscalar(original):
            assert abs(original-(menu['native'][0]-menu['native'][1]))<1e-8
            original=menu['native']
    else:ob,cfg,est,prev,cmd,original=restore_old(card,mod)
    assert ob['player']==card['seat'] and ob['step']==card['step'] and 'full_carbon' not in ob
    assert cmd[0]!=cmd[1] and len(original)==2
    dry=simulate(mod,ob,cfg,est,prev,cmd,[card['native_draw_seed']])
    err=float(np.max(np.abs(np.array(dry['values'][0])-original)))
    row.update(native_values=original,native_reproduced=dry['values'][0],reproduction_max_abs=err,
               commands=cmd,estimated_carbon_sha=sha(np.asarray(est,dtype='<f8').tobytes()),native_dry=dry)
    if err>2e-4:
        return row|dict(status='reconstruction_failed',reason='H8 archived reproduction outside absolute 2e-4')
    multi=simulate(mod,ob,cfg,est,prev,cmd,card['draw_seeds'])
    vals=np.array(multi['values']);delta=vals[:,1]-vals[:,0];mu=float(np.mean(delta))
    se=float(np.std(delta,ddof=1)/math.sqrt(len(delta)))
    incumbent=int(original[1]>original[0]);choice=incumbent if mu==0 else int(mu>0)
    row.update(status='complete',draw_values=multi['values'],simulated_opponent_orders=multi['opponent_root'],
               unique_enemy_orders=len({json.dumps(x,sort_keys=True) for x in multi['opponent_root']}),
               sim_terminal_flags=multi['exact'],simulated_final_state_steps=multi['final_steps'],
               offline_cpu=multi['cpu'],offline_wall=multi['wall'],choice_native=incumbent,choice_mean=choice,
               choice_fresh_single=incumbent if delta[0]==0 else int(delta[0]>0),mean_delta_B_minus_A=mu,
               conditional_mc_se=se,conditional_t7_95=[mu-2.36462425*se,mu+2.36462425*se])
    return row

def gather(groups):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    models={}
    for card in PLAN['tickets']:
        if card['group'] not in groups:continue
        p=HERE/'predictions'/(key(card)+'.json')
        if p.exists():continue
        file=D/('ladder/prediction_feedback/package/candidate.py' if card['group']=='close' else 'cold_start/candidate.py')
        assert sha(file.read_bytes())==card['parent']
        if card['parent'] not in models:models[card['parent']]=from_native(file)
        row=evaluate(card,models[card['parent']])
        temp=p.with_suffix('.tmp');temp.write_text(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');temp.replace(p)
        print(key(card),row['status'],row.get('reproduction_max_abs'),row.get('choice_mean'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',nargs='+',choices=['train','held','close']);a=p.parse_args()
    assert len(os.sched_getaffinity(0))==1
    gather(a.group)
