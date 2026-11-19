"""Show or restore the registered native supervisors on Linux, without releasing their experiment locks.

Run with the experiment interpreter: python native_progress.py status|resume [JOB ...].
Jobs share a Linux checkout and its existing flock files. Status only reads state.
--cache-dir (default: XDG_CACHE_HOME or ~/.cache) holds per-checkout launch locks and append-only logs.
Only the existing registered controllers decide which maps/arms may start.
"""
from dataclasses import dataclass
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

BASE = Path(__file__).resolve().parent
TRAIN = BASE / 'tree_response_training'
SUP = TRAIN / 'supplement_01'
HEAD = TRAIN / 'response_head'
EARLY = BASE / 'early_lane_dev/native_response'
ECON = BASE / 'economy_aux_dev/native_dev'
QUEUE = BASE / 'endgame_queue_dev'
HIRE = BASE / 'late_workforce_dev'


@dataclass(frozen=True)
class Job:
    name: str
    script: Path
    action: str
    lock: Path
    # Known lower-level owners may survive a dead supervisor.
    children: tuple = ()


JOBS = {
    j.name:j for j in (
        Job('training', SUP/'run.py', 'drain', SUP/'.owner.lock',
            ('tree_response_training/supplement_01/batches/*/.intake.schedule.lock',
             'tree_response_training/supplement_01/batches/*/source/.*lock',
             'tree_response_training/supplement_01/batches/*/case_*/.response_*.lock')),
        Job('response_head', HEAD/'finish_registered.py', 'wait', HEAD/'.handoff-owner.lock',
            ('tree_response_training/response_head/development_runs/*/.lock',)),
        Job('early_lane', EARLY/'run.py', 'schedule', EARLY/'.schedule.lock',
            ('early_lane_dev/native_response/case_*/.parent.lock','early_lane_dev/native_response/case_*/.alternative.lock')),
        Job('economy', ECON/'run.py', 'run', ECON/'.owner.lock', ('economy_aux_dev/native_dev/*/.lock',)),
        Job('economy_report', ECON/'report.py', '--wait', ECON/'.attribution.lock'),
        Job('endgame_queue', QUEUE/'native_last.py', '--wait', QUEUE/'.native_supervise.lock'),
        Job('late_hiring', HIRE/'native.py', 'schedule', HIRE/'native/.schedule.lock',
            ('late_workforce_dev/native/case_*/.parent.lock', 'late_workforce_dev/native/case_*/.alternative.lock')),
    )
}


def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, detail):
    if not ok: raise ValueError(detail)


def check_binding(obj, key, path):
    require(obj[key]==digest(path), f'{key} does not match {path.name}')


def endgame_progress():
    plan = read(QUEUE/'native_protocol.json')
    wanted = {(r['seed'], r['seat']) for r in plan['cases']}
    require(len(wanted) == len(plan['cases']) == 4, 'missing registered terminal case')
    result = QUEUE/'native_result.json'
    grant = SUP/'endgame_queue_grant.json'
    if grant.exists():
        permit = read(grant)
        check_binding(permit, 'completion_sha256', SUP/'completion.json')
        check_binding(permit, 'last_move_protocol_sha256', QUEUE/'native_protocol.json')
        require(permit['cpu'] == 0 and permit['permitted_native_calls'] == 8, 'wrong final-turn allocation')
    if not result.exists():
        return 'ready', {'phase': 'last_turns_granted' if grant.exists() else 'awaiting_intake_08_court',
                         'signed_native_arms': len(list(QUEUE.glob('last_arm_*.json.gz')))}
    doc = read(result)
    for k, p in [('protocol_sha256', QUEUE/'native_protocol.json'), ('grant_sha256', grant),
                 ('completion_sha256', SUP/'completion.json'), ('runner_sha256', QUEUE/'native_last.py')]:
        check_binding(doc, k, p)
    require(doc['candidate_sha256'] == plan['candidate_sha256'] and doc['examiner_sha256'] == plan['source_examiner'],
            'terminal player identity differs')
    require(len(doc['pairs']) == 4 and {(r['seed'],r['seat']) for r in doc['pairs']} == wanted and
            doc['native_calls'] == 8 and len(doc['pair_sha256']) == 4, 'incomplete terminal evidence')
    for seed, seat in wanted:
        path = QUEUE/f'native_{seed}_{seat}.json.gz'
        check_binding(doc['pair_sha256'], path.name, path)
        case = read(path)
        require(case['protocol_sha256'] == doc['protocol_sha256'] and (case['seed'],case['seat']) == (seed,seat)
                and set(case['arms']) == {'v1','candidate'}
                and all(a['end']['status'] == ['DONE','DONE'] and a['search_call_count'] == 1 for a in case['arms'].values()),
                'missing terminal callback or official score')
    return 'complete', {'phase':'signed_original_last_turns', 'pairs':4, 'native_last_turn_calls':8}


def hiring_progress():
    out = HIRE/'native'
    plan = read(out/'protocol.json')
    wanted = {(r['seed'], r['seat'], r['t']) for r in plan['cases']}
    require(len(wanted) == len(plan['cases']) == 2, 'missing registered hiring case')
    result = out/'results.json'
    if not result.exists():
        return 'ready', {'phase': 'live_responses_granted' if (out/'granted.json').exists() else 'awaiting_registered_predecessors',
                         'signed_native_arms': sum(1 for label in ('parent','alternative') for _ in out.glob(f'case_*/live_{label}.json'))}
    doc, grant = read(result), read(out/'granted.json')
    check_binding(doc,'protocol_sha256',out/'protocol.json')
    check_binding(grant,'protocol_sha256',out/'protocol.json')
    check_binding(grant,'shared_final_probe_sha256',QUEUE/'native_result.json')
    check_binding(grant,'tree_head_acceptance_sha256',HEAD/'independent_acceptance.json')
    require(doc['candidate_sha256'] == plan['candidate_sha256'] and doc['source_parent_sha256'] == plan['parent_sha256']
            and doc['all_four_responses'] is True and doc['formal_games'] == 0 and len(doc['native_cases']) == 2
            and {(r['seed'],r['seat'],r['turn']) for r in doc['native_cases']} == wanted, 'incomplete hiring comparison')
    for row in doc['native_cases']:
        case = out/f"case_{row['seed']}_{row['seat']}"
        check_binding(row, 'proof_sha256', case/'verified.json.gz')
        proof = read(case/'verified.json.gz')
        require(set(proof['official']) == {'parent','alternative'}, 'missing independent hiring score')
        for arm, score in proof['official'].items():
            native = read(case/f'live_{arm}.json')
            check_binding(native, 'action_sha256', case/native['action_tape'])
            require(score['statuses'] == native['statuses'] == ['DONE','DONE'] and
                    score['score_by_physical_seat'] == native['score'] and score['raw_sha256'] == native['action_sha256']
                    and score['verified_native_search_calls'] == native['search_calls'] > 0,
                    'hiring score differs from signed replays')
    return 'complete', {'phase':doc['decision'], 'pairs':2, 'native_arms':4}


def summary(name):
    """Read native receipts. Return (phase, detail); a preliminary score is never a terminal seal."""
    if name=='endgame_queue': return endgame_progress()
    if name=='late_hiring': return hiring_progress()
    if name=='training':
        finish=SUP/'completion.json'
        if finish.is_file():
            doc=read(finish)
            require(doc['phase'] in ('registered_count_target_reached','all_registered_blocks_exhausted') and doc['next'] is None,'unfinished training phase in completion')
            check_binding(doc,'registration_sha256',SUP/'registration.json')
            for batch in doc['counts']['source_intakes']:
                check_binding(batch,'labels_sha256',TRAIN/batch['dir']/'training_labels.json')
            return 'complete',{'phase':doc['phase'],'paid_pairs':doc['counts']['actual_paid_pairs']}
        blocks=sorted((SUP/'batches').glob('intake_*'))
        if not blocks: return 'ready',{'phase':'waiting_for_registered_source'}
        latest=blocks[-1]
        progress=latest/'source/results.json'
        return 'ready',{'intake':latest.name,'native_source_games':len(read(progress)['games']) if progress.exists() else 0,
                        'finished_response_arms':sum(1 for arm in ('parent','alternative') for _ in latest.glob(f'case_*/live_{arm}.json')),
                        'latest_aggregate':(latest/'aggregate.json').exists()}
    if name=='response_head':
        acc=HEAD/'independent_acceptance.json'
        if not acc.exists(): return 'ready',{'phase':'waiting_for_independent_acceptance'}
        doc=read(acc)
        for k,p in [('completion_sha256',SUP/'completion.json'),('status_sha256',HEAD/'status.json'),('sample_tape_sha256',HEAD/'samples.json.gz')]:
            check_binding(doc,k,p)
        require(isinstance(doc['passed'],bool) and read(HEAD/'status.json')['eligible_for_development'] is doc['passed'],'head eligibility differs')
        if not doc['passed']: return 'complete',{'phase':'heldout_gate_closed','reason':doc['reason'],'development_games':0}
        directory=HEAD/'development_runs';comparison=directory/'comparison.json'
        if not comparison.exists():return 'ready',{'phase':'eight_registered_development_games_required'}
        result=read(comparison)
        check_binding(result,'release_sha256',directory/'release.json')
        release=read(directory/'release.json')
        maps=release['maps']
        check_binding(release,'development_protocol',HEAD/'protocol.json')
        check_binding(release,'source_protocol',TRAIN/'protocol.json')
        require(release['master3s2_sha256']==release['arena_sources']['data/master3s2_source.py'], 'examiner is not the registered original')
        expected={(s,p) for s in maps for p in (0,1)}
        paired={(r['seed'],r['seat']):r for r in result['pairs']}
        require(len(expected)==len(result['pairs'])==4 and set(paired)==expected and result['games']==8 and result['formal'] is False,'incomplete head comparison')
        for arm in ('parent','candidate'):
            native=read(directory/arm/'results.json')
            replay=read(directory/arm/'official_replay.json')
            official=replay['runs'][arm]
            header=native['header']
            package=release['packages'][arm]
            require(native['status']=='complete' and len(native['games'])==len(official['games'])==4
                    and official['pack_sha256']==header['pack_sha256']==package['pack_sha256']
                    and header['weight_sha256']==package['weight_sha256'] and header['seeds']==maps
                    and header['cpus']==release['cpus'] and header['source_sha256']==release['arena_sources']
                    and replay['engine_source_sha256']==release['official_engine'], 'incorrect or unfinished native arm')
            by_key={(r['seed'],r['candidate_seat']):r for r in native['games']}
            require(set(by_key)=={(g['seed'],g['seat']) for g in official['games']}==expected,'duplicate or missing held-out map/seat')
            for g in official['games']:
                key=g['seed'],g['seat'];r=by_key[key]
                check_binding(r,'raw_sha256',directory/arm/r['raw_file'])
                require(r['raw_sha256']==g['transcript_sha256'] and g['statuses']==['DONE','DONE'] and
                        g['map_sha256']==release['boards'][str(key[0])] and g['steps']==r['actual_transitions']
                        and r['result']==g['result'] and r['official_score_by_seat']==g['official_reward_by_seat']
                        and all(paired[key][arm][k]==v for k,v in g.items()),'mismatched official trajectory')
        return 'complete',{'phase':result['verdict'],'development_games':8}
    early=name=='early_lane'
    slot=SUP/('early_lane_slot' if early else 'economy_slot')
    released,skipped=slot/'released.json',slot/'skipped.json'
    require(not (released.exists() and skipped.exists()),'conflicting release and skip')
    if skipped.exists():
        doc=read(skipped); study=EARLY if early else ECON
        check_binding(doc,'protocol_sha256',study/'protocol.json')
        check_binding(doc,'skip_sha256',(EARLY.parent/'early_lane_skip.json') if early else ECON/'mechanism_skip.json')
        require(doc['native_arms' if early else 'signed_games']==0,'not a zero-game skip')
        return 'complete',{'phase':'registered_skip'}
    if released.exists():
        seal=read(released);check_binding(seal,'grant_sha256',slot/'granted.json')
        path=(EARLY/'result.json') if early else (ECON/'comparison.json')
        check_binding(seal,'result_sha256' if early else 'comparison_sha256',path)
        count=seal['original_responses' if early else 'signed_games']
        require(count==(8 if early else 12),'unexpected number of signed responses')
        result=read(path)
        if early:
            require(result['status']=='complete' and result['completed_pairs']==4 and result['completed_arms']==count,'unfinished live branches')
            require(len(seal['cases'])==len({(r['seed'],r['seat']) for r in seal['cases']})==4,'missing or repeated certified root')
            for row in seal['cases']:
                case=EARLY/f"case_{row['seed']}_{row['seat']}"
                check_binding(row,'proof_sha256',case/'verified.json.gz')
        else:
            require(set(seal['arms'])=={'parent','control','candidate'} and len(result['paired_map_seats'])==4, 'incomplete economic evidence')
            for arm,proof in seal['arms'].items():
                check_binding(proof,'results_sha256',ECON/arm/'results.json')
                check_binding(proof,'official_sha256',ECON/arm/'official_replay.json')
        if name=='economy_report':
            tape=ECON/'actual_attribution.json.gz';report=ECON/'RESULTS.md'
            if not (tape.exists() and report.is_file()):return 'ready',{'phase':'writing_economic_attribution'}
            proof=read(tape);check_binding(proof,'native_release_sha256',released);check_binding(proof,'comparison_sha256',path)
            require(len(proof['games'])==4 and report.stat().st_size>0,'incomplete attribution')
        return 'complete',{'phase':'released','signed_games':count}
    if name=='economy_report':return 'waiting',{'phase':'waiting_for_12_signed_games'}
    if early and not (EARLY.parent/'parity.json').exists():return 'waiting',{'phase':'waiting_for_actual_package_parity'}
    return 'ready',{'phase':'native_slot_granted' if (slot/'granted.json').exists() else 'waiting_for_native_slot'}


def lock_held(path):
    """Status must not create, truncate or remove the owner's lock file."""
    try:handle=path.open('rb')
    except FileNotFoundError:return False
    with handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        return False


def matching_pids(job, proc=Path('/proc')):
    wanted=job.script.resolve()
    result=[]
    for entry in proc.glob('[0-9]*'):
        try:
            state=(entry/'stat').read_text().rsplit(')',1)[1].split()[0]
            if state in ('Z','X','x'):continue
            args=[os.fsdecode(a) for a in (entry/'cmdline').read_bytes().split(b'\0') if a]
            if len(args)<3 or not Path(args[0]).name.startswith('python') or args[-1]!=job.action:continue
            if any(a not in ('-u','-B','-E','-s','-S','-I','-O','-OO') for a in args[1:-2]):continue
            script=Path(args[-2])
            if not script.is_absolute():script=(entry/'cwd').resolve(strict=True)/script
            if script.resolve()==wanted:result.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            # A process may exit at any point while the status snapshot is read.
            continue
    return sorted(result)



def within_startup(pids, proc, seconds):
    """Allow an external launcher to enter its own lock; a long-lived PID alone is not an owner."""
    if not pids:
        return False
    uptime = float((proc/'uptime').read_text().split()[0])
    ticks = os.sysconf('SC_CLK_TCK')
    for pid in pids:
        try:
            fields = (proc/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
            if uptime - int(fields[19])/ticks < seconds:
                return True
        except FileNotFoundError:
            continue
    return False

def active_children(job):
    return [str(p.relative_to(BASE)) for glob in job.children for p in BASE.glob(glob) if lock_held(p)]


def ensure_owner(job,cache,*,resume=False,proc=Path('/proc'),startup_timeout=10):
    launch=cache/(job.name+'.launch.lock')
    def inspect():
        pids=matching_pids(job,proc)
        held=lock_held(job.lock)
        phase='active' if held else 'starting' if lock_held(launch) or within_startup(pids,proc,startup_timeout) else 'idle'
        info = {'state':phase,'native_lock_held':held,'matching_live_pids':pids}
        # An owner may predate this tool. Its output is useful without treating its PID as authority.
        if held:
            for pid in pids:
                try:
                    stdout=(proc/str(pid)/'fd/1').resolve(strict=True)
                    if stdout.is_file():
                        info['log']=str(stdout)
                        break
                except (FileNotFoundError,PermissionError):
                    continue
        if 'log' not in info and (cache/(job.name+'.log')).is_file():
            info['last_managed_log']=str(cache/(job.name+'.log'))
        return info
    before=inspect()
    if not resume or before['state']!='idle':return before
    cache.mkdir(parents=True,exist_ok=True)
    # Keep this separate lock across fork/exec, including startup and orphan-worker drainage.
    # Closing this process's fd leaves the detached child's fd as sole owner without a PID file.
    with launch.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'state':'starting','native_lock_held':lock_held(job.lock),'matching_live_pids':matching_pids(job,proc)}
        if lock_held(job.lock) or within_startup(matching_pids(job,proc),proc,startup_timeout):return inspect()
        # A concurrent controller could have sealed the study between the initial status and dispatch.
        if job.name in JOBS:
            phase,detail=summary(job.name)
            if phase!='ready':return {'state':phase,'detail':detail}
        orphans=active_children(job)
        command=([sys.executable,'-u',str(Path(__file__).resolve()),'_drain',job.name]
                 if orphans else [sys.executable,'-u',str(job.script),job.action])
        log=cache/(job.name+'.log')
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        with log.open('ab',buffering=0) as output:
            worker=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                                    cwd=job.script.parent,start_new_session=True,pass_fds=(lock.fileno(),),env=env)
        deadline=time.monotonic()+startup_timeout
        while not orphans and not lock_held(job.lock) and worker.poll() is None and time.monotonic()<deadline:
            time.sleep(.03)
        if worker.poll() not in (None,0):raise RuntimeError(f'{job.name} failed to launch; examine {log}')
        if worker.returncode is None:
            # Long-lived callers reap detached children without blocking a status/resume call.
            threading.Thread(target=worker.wait,daemon=True).start()
        return {'state':'started','pid':worker.pid,'native_lock_held':lock_held(job.lock),'log':str(log),'surviving_worker_locks':orphans}


def drain_orphans(job):
    while active_children(job):
        time.sleep(2)
    # Natural child exit can itself be the completion boundary.
    if summary(job.name)[0]!='ready' or lock_held(job.lock):return
    os.execv(sys.executable,[sys.executable,'-u',str(job.script),job.action])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('status','resume','_drain'),metavar='{status,resume}')
    ap.add_argument('jobs',nargs='*',metavar='JOB',help='default: all of '+', '.join(JOBS))
    ap.add_argument('--cache-dir',type=Path,help='explicit runtime log/dispatch-lock directory')
    args=ap.parse_args()
    if any(name not in JOBS for name in args.jobs):ap.error('unknown job; choose '+', '.join(JOBS))
    if args.action=='_drain':
        if len(args.jobs)!=1:ap.error('one job required')
        return drain_orphans(JOBS[args.jobs[0]])
    cache=args.cache_dir or (Path(os.environ.get('XDG_CACHE_HOME') or Path.home()/'.cache')/'carbon-league/native-progress'/hashlib.sha256(os.fsencode(BASE)).hexdigest()[:16])
    failed=False;results={}
    for name in args.jobs or JOBS:
        try:
            phase,detail=summary(name);job=JOBS[name]
            if phase=='complete':record={'state':phase}
            else:
                record=ensure_owner(job,cache,resume=args.action=='resume' and phase=='ready')
                if record['state']=='idle' and phase=='waiting':record['state']='waiting'
            results[name]={**record,'detail':detail}
        except (KeyError,ValueError,RuntimeError,OSError) as exc:
            results[name]={'state':'error','reason':str(exc)};failed=True
    print(json.dumps(results,ensure_ascii=False,indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
