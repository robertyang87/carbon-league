"""Isolated two-turn purchase, followed by ordinary v1 and fully responding original Master3s2."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from build import BASE,HERE,PLAN,ROOT,SOURCE,sha,source
from screen import read,retain
sys.path.insert(0,str(BASE/'root_recovery_branches'))
import live
from branches import clean
import numpy as np

def check_previous_queue():
    fit=HERE.parent/'terminal_return_fit'
    mm=read(fit/'multimap/development.json')
    other=read(fit/'opponent_transfer/results.json')
    if (not mm.get('ran') and mm.get('reason')!='training gate retained byte-identical v1; redundant games skipped'):
        raise ValueError('waiting on multimap conditional development')
    if any(len(other['games'][a])!=4 for a in ('parent','learned')):
        raise ValueError('prior transfer incomplete')

def setup():
    plan=read(PLAN);gate=read(HERE/'followup_gate.json');mech=read(HERE/'mechanism.json')
    if (not mech['both_preassigned_roots'] or sha(source())!=mech['package_sha256'] or
            gate['root_protocol_sha256']!=sha(PLAN.read_bytes()) or gate['selected_evidence_sha256']!=sha((HERE/'mechanism.json').read_bytes())
            or sha((HERE/'agent.py').read_bytes())!=gate['package_sha256']):
        raise ValueError('changed policy, selection or registration')
    values=read(HERE/'value_evaluation.json')
    for name in ('test_seen_pool','test_unseen_opponent'):
        q=values['conditional_measures'][name]['conditional_quartiles']
        if not q['top']['purchases'] or not q['bottom']['purchases'] or q['top']['mean_realized']<=q['bottom']['mean_realized']:
            raise ValueError('value ordering gate failed')
    if sha((HERE/'value_weights.json').read_bytes())!=values['weights_sha256']:
        raise ValueError('evaluated value head does not match')
    native=subprocess.check_output(['git','show',plan['source_revision']+':sources/master3s2/master3s2.py'],cwd=ROOT)
    if sha(native)!=live.MASTER_SHA or sha(live.SOURCE.read_bytes())!=live.MASTER_SHA:
        raise ValueError('not the original opponent')
    if sha(SOURCE.read_bytes())!=plan['parent_pack_sha256']:
        raise ValueError('changed v1')
    live.PACK=SOURCE; live.PACK_HASH=plan['parent_pack_sha256']
    check_previous_queue()
    return mech['selected_roots']

def run_one(idx,arm):
    roots=setup(); reg=roots[idx];seat=reg['seat'];seed=reg['seed']
    directory=HERE/f'response_{seed}_{seat}';directory.mkdir(exist_ok=True)
    obs=read(HERE/reg['root']);native=read(ROOT/reg['source'])
    if (sha((HERE/reg['root']).read_bytes())!=reg['root_sha256'] or sha((ROOT/reg['source']).read_bytes())!=reg['source_sha256']
            or obs['root_parent_joint']!=clean(native['actions'][reg['t']][seat])
            or obs['root_option_joint']!={k:v for k,v in obs['root_parent_joint'].items() if k!=reg['worker']}
            or obs['root_opponent_fixed']!=native['actions'][reg['t']][1-seat]):
        raise ValueError('root or original joint differs')
    root={'t':reg['t'],'worker':reg['worker'],'old':reg['old'],'alternative':'NONE','tree':obs['ticket']['tree'] or ''}
    case={'seed':seed,'seat':seat,'family':'two_turn_planter_exercise','root_t':root['t'],'source_dir':str(Path(reg['source']).parent),
          'record':{'seed':seed,'seat':seat,'source':reg['source'],'source_sha256':reg['source_sha256']},
          'root':root,'actual_option_source':read(HERE/'manifest.json')['pack_sha256'],
          'root_sha256':reg['root_sha256'],'response_code_sha256':sha(Path(live.__file__).read_bytes()),
          'followup_sha256':sha((HERE/'followup_gate.json').read_bytes())}
    with (directory/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        retain(directory/'live_protocol.json',case)
        shared=directory/'live_shared_init.json'
        rng=np.random.RandomState(reg['paired_rng_seed']).get_state()
        common={'mt_algorithm':rng[0],'mt_keys':rng[1].tolist(),'mt_index':rng[2],'has_normal':rng[3],
                'stored_normal':rng[4],'source_sha256':live.MASTER_SHA}
        if not shared.exists():
            master=live.module(live.SOURCE,'warm_original')
            retain(shared,{**common,'warm_cost':list(master._S_COST)})
        if any(read(shared).get(k)!=v for k,v in common.items()):
            raise ValueError('branch-paired random stream changed')
    live.run(arm,directory,case,resume=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('cmd',choices=('launch','run'));p.add_argument('--root',type=int);p.add_argument('--arm');a=p.parse_args()
    if a.cmd=='run':
        if a.root not in (0,1) or a.arm not in ('parent','alternative'):
            raise ValueError('unregistered arm')
        run_one(a.root,a.arm);return
    roots=setup()
    jobs=[(idx,arm) for idx in range(len(roots)) for arm in ('parent','alternative')]
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def launch(item):
        cpu,(idx,arm)=item
        cmd=['taskset','-c',str(cpu),sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(idx),'--arm',arm]
        subprocess.run(cmd,env=env,check=True)
    with (HERE/'.response-schedule.lock').open('a') as lock, ThreadPoolExecutor(max_workers=4) as pool:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        list(pool.map(launch,enumerate(jobs)))

if __name__=='__main__':
    main()
