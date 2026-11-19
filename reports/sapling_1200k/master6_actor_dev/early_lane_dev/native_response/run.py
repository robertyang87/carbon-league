"""Resume four frozen, single-intervention terminal comparisons using both original policies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
LANE=HERE.parent
TRIAL=LANE.parent/'tree_response_training'
SAPLING=LANE.parents[1]
ROOT=SAPLING.parents[1]
sys.path.insert(0,str(SAPLING/'root_recovery_branches'))
import live

PLAN_SHA='1a461718ed0e0e161838cc5bde1592ed34cb66bde94893d135b3d5bd4b023db8'
ENV={**os.environ, **{v:'1' for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
PARENT=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'


def require(ok, detail):
    if not ok: raise ValueError(detail)


def read(path): return json.loads(path.read_bytes())

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def seal(path,data):
    if path.exists():
        require(read(path)==data,f'frozen file changed: {path}')
    else:
        live.save(path,data)


def utc(): return datetime.now(timezone.utc).isoformat()


def identity():
    plan=read(HERE/'protocol.json')
    require(sha(HERE/'protocol.json')==PLAN_SHA and sha(LANE/'candidate.py')==plan['candidate_sha256']
            and sha(PARENT)==plan['compared_parent_sha256'] and sha(live.SOURCE)==plan['examiner_sha256']==live.MASTER_SHA
            and sha(Path(live.__file__))==plan['live_responder_sha256'],'frozen actor, original examiner or runner changed')
    for file,field in [('gate.json','gate_sha256'),('response_plan.json','roots_file_sha256'),('protocol.json','protocol_sha256'),('weights.json','weights_sha256')]:
        require(sha(LANE/file)==plan[field],f'changed upstream {file}')
    parity=read(LANE/'parity.json'); gate=read(LANE/'gate.json')
    require(parity['result']=='pass' and parity['gate_sha256']==plan['gate_sha256'] and parity['candidate_sha256']==plan['candidate_sha256']
            and parity['total']['all_order_agreements']==gate['overall']['source_turns'] and
            parity['total']['overrides_agree']==gate['overall']['issued'] and
            len(parity['evidence'])==len(gate['covered_artifacts'])==8,'incomplete real entry-point parity')
    for label,digest in parity['evidence'].items():
        require(sha(LANE/'parity_rows'/f"{label.replace(':','_')}.json")==digest,'lost original root parity')
    expected=[(s['seed'],s['seat'],s['bins']['issued']['signature']) for s in read(LANE/'response_plan.json')['roots'] if s['seed'] in (621474120,621474121)]
    require([(r['seed'],r['seat'],r['signature']) for r in plan['roots']]==expected and len(set(expected))==4,'wrong signed roots')
    return plan


def directory(row):return HERE/f"case_{row['seed']}_{row['seat']}"


def prepare(plan,row):
    wd=directory(row);wd.mkdir(exist_ok=True)
    path=ROOT/row['source']; require(sha(path)==row['source_sha256'],'original tape changed')
    link=read(LANE/'parity_rows'/f"{row['seed']}_{row['seat']}.json")
    expected={'uid':row['own_uid'],'order':row['alternative_order'],'old':row['parent_order'],'t':row['t'],'signature':row['signature']}
    require([r for r in link['changes'] if r['t']==row['t']]==[expected],'single package did not really issue registered intervention')
    q=dict(seed=row['seed'],seat=row['seat'],family='early_lane_paired_response_01',root_t=row['t'],source_dir=str(path.parent.relative_to(ROOT)),
           record={k:row[k] for k in ('seed','seat','source','source_sha256')},
           root=dict(t=row['t'],worker=row['own_uid'],enemy_worker=row['target_uid'],old=row['parent_order'],alternative=row['alternative_order'],
                     corridor=row['corridor'],tree='',signature=row['signature']),
           study_protocol_sha256=PLAN_SHA,source_parity_sha256=sha(LANE/'parity.json'),real_candidate_sha256=plan['candidate_sha256'],
           common_continuation_sha256=plan['compared_parent_sha256'],responder_sha256=plan['live_responder_sha256'])
    seal(wd/'live_protocol.json',q)
    rand=np.random.RandomState(row['new_stream_seed']).get_state()
    fixed=dict(mt_algorithm=rand[0],mt_keys=rand[1].tolist(),mt_index=rand[2],has_normal=rand[3],stored_normal=rand[4],source_sha256=live.MASTER_SHA)
    shared=wd/'live_shared_init.json'
    if not shared.exists():
        opp=live.module(live.SOURCE,'initial_native_master3s2')
        fixed['warm_cost']=list(opp._S_COST)
        require(opp._SEED is None and opp._TEMP==1 and opp._SEARCH_OK and opp._RX and fixed['warm_cost'],'wrong examiner')
        seal(shared,fixed)
    else:
        old=read(shared)
        require(all(old[k]==v for k,v in fixed.items()) and old['warm_cost'],'shared RNG or warm-start cost changed')
    return wd,q


def read_env():
    processes=[]
    for p in Path('/proc').glob('[0-9]*'):
        try: cmd=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if not cmd or b'python' not in cmd[0].lower():continue
        paths=[os.fsdecode(s) for s in cmd if s.endswith(b'.py')]
        if any(n in v for n in ('tree_pack_arena','symmetry_dev/run','pair_games','supplement_01/run.py','native_response/run.py') for v in paths):
            processes.append({'pid':int(p.name),'scripts':paths,'cpu_affinity':sorted(os.sched_getaffinity(int(p.name)))})
    return dict(at_utc=utc(),cgroup_cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                cgroup_stat=Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines(),loadavg=Path('/proc/loadavg').read_text().strip(),native_processes=processes)


def worker(i,arm,cpu):
    require(os.sched_getaffinity(0)=={cpu},'wrong per-arm CPU affinity')
    plan=identity(); row=plan['roots'][i];wd=directory(row)
    execution=read(HERE/'execution.json')
    require(execution['run_sha256']==sha(Path(__file__)) and execution['verify_sha256']==sha(HERE/'verify.py'),'runner or outcome audit changed')
    require((TRIAL/'supplement_01/early_lane_slot/granted.json').is_file(),'06/07 handoff missing')
    require(sha(wd/'live_protocol.json')==execution['roots'][str(i)]['case_sha256'] and
            sha(wd/'live_shared_init.json')==execution['roots'][str(i)]['shared_sha256'],'detached pair inputs')
    with (wd/f'.{arm}.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        live.PACK=PARENT;live.PACK_HASH=plan['compared_parent_sha256']
        live.run(arm,wd,read(wd/'live_protocol.json'),resume=True)


def execution_proof(plan):
    rows={}
    for i,r in enumerate(plan['roots']):
        case,_=prepare(plan,r)
        rows[str(i)]={'case_sha256':sha(case/'live_protocol.json'),'shared_sha256':sha(case/'live_shared_init.json')}
    blob=dict(protocol_sha256=PLAN_SHA,parity_sha256=sha(LANE/'parity.json'),run_sha256=sha(Path(__file__)),
              verify_sha256=sha(HERE/'verify.py'),root_helper_sha256=sha(Path(live.__file__).parent/'branches.py'),
              shared_ledger_sha256=sha(SAPLING/'planter_routes_dev/endgame_responses/verify.py'),roots=rows)
    seal(HERE/'execution.json',blob)
    return blob


def report(plan,verified):
    metrics=[r['metrics'] for r in verified]
    maps={}
    for idx in (0,2):
        m=plan['roots'][idx]['seed'];pair=metrics[idx:idx+2]
        maps[str(m)]={k:mean(x[k] for x in pair) for k in ('delta_u','delta_margin','target_banked10_delta','own_team_banked10_delta')}
    avg={k:mean(r[k] for r in maps.values()) for k in next(iter(maps.values()))}
    tests={'mean_u_positive_on_each_terrain':all(r['delta_u']>0 for r in maps.values()),
           'target_worker_ten_turn_own_bank_down':avg['target_banked10_delta']<0,
           'our_team_ten_turn_own_bank_preserved':avg['own_team_banked10_delta']>=0}
    files={}
    for r in plan['roots']:
        wd=directory(r)
        for q in ('live_protocol.json','live_shared_init.json','resource_at_start.json','verified.json.gz')+tuple(
                f'live_{a}{s}' for a in ('parent','alternative') for s in ('.json','_actions.json.gz')):
            files[str((wd/q).relative_to(HERE))]=sha(wd/q)
    files['execution.json']=sha(HERE/'execution.json')
    cert=dict(status='complete',completed_pairs=len(verified),completed_arms=2*len(verified),protocol_sha256=PLAN_SHA,
              candidate_sha256=plan['candidate_sha256'],parent_sha256=plan['compared_parent_sha256'],examiner_sha256=plan['examiner_sha256'],
              parity_sha256=sha(LANE/'parity.json'),files_sha256=files,by_case=metrics,by_map=maps,map_equal=avg,conditions=tests,
              decision='replicate_on_new_development_positions' if all(tests.values()) else 'stop_this_construction')
    seal(HERE/'result.json',cert)
    return cert


def schedule():
    plan=identity()
    verify_case=live.module(HERE/'verify.py','signed_early_lane_verifier').verify_case
    # Set up the immutable per-case RNG/weights outside the critical assignment. Never start a native episode here.
    with (HERE/'.schedule.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execution_proof(plan)
        sys.path.insert(0,str(TRIAL))
        sys.path.insert(0,str(TRIAL/'supplement_01'))
        import early_lease
        print('waiting for exclusive window after completed intake 06',flush=True)
        with early_lease.native_session():
            verified=[]
            for i,row in enumerate(plan['roots']):
                wd=directory(row);arms=('parent','alternative') if i%2==0 else ('alternative','parent')
                if not (wd/'resource_at_start.json').exists():
                    seal(wd/'resource_at_start.json',{**read_env(),'assigned_cpus':dict(zip(arms,(0,1))),'study_protocol_sha256':PLAN_SHA})
                def launch(entry):
                    cpu,arm=entry
                    subprocess.run(['taskset','-c',str(cpu),sys.executable,'-u',str(Path(__file__).resolve()),'worker',str(i),arm,str(cpu)],env=ENV,check=True)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(launch,enumerate(arms)))
                v=verify_case(wd);verified.append(v)
                print('paired response verified',row['seed'],row['seat'],v['metrics'],flush=True)
            report(plan,verified)
        early_lease.release()


if __name__=='__main__':
    args=sys.argv[1:]
    if args==['schedule']:schedule()
    elif args==['prepare']:execution_proof(identity())
    elif len(args)==4 and args[0]=='worker':worker(int(args[1]),args[2],int(args[3]))
    else:raise SystemExit('usage: run.py prepare | schedule | worker INDEX parent/alternative CPU')
