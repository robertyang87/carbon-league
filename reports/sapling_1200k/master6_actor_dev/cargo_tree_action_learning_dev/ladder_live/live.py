"""Registered live dual-container match; fail closed before invoking a policy on an ineligible host."""
import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import fcntl
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
REGISTRATION = HERE / 'protocol.json'
CLAIM = HERE / 'attempt.json'
GAME_LOCK = HERE.parents[1] / 'tree_response_training/.continuation.lock'
PLAN = json.loads(REGISTRATION.read_text())
RULE = PLAN['platform']
ROLES = ('opponent','candidate')  # physical seats 0, 1


def configure(registration, claim, lock):
    """Select an immutable match contract before creating either player process."""
    global PLAN, REGISTRATION, CLAIM, GAME_LOCK, RULE, ROLES
    plan = json.loads(registration.read_bytes())
    seat = plan['candidate_seat']
    if type(seat) is not int or seat not in (0, 1) or plan['opponent_seat'] != 1 - seat:
        raise ValueError('invalid physical seats')
    PLAN, REGISTRATION, CLAIM, GAME_LOCK = plan, registration, claim, lock
    RULE = plan['platform']
    ROLES = ('candidate', 'opponent') if seat == 0 else ('opponent', 'candidate')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, separators=(',',':'), allow_nan=False).encode() + b'\n'


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()+b'\n')
    temp.replace(path)


def eligibility(probes, container_settings):
    issues = []
    for seat, p in enumerate(probes):
        who = ROLES[seat]
        c = p['cgroup']
        quota = (c['cpu.max'] or '').split()
        if len(quota)!=2 or not all(x.isdecimal() for x in quota) or int(quota[0])*2 != int(quota[1])*3 or int(quota[1])<=0:
            issues.append(f'{who}: cpu.max={c["cpu.max"]}, expected 1.5 cores')
        if c['memory.max']!=str(RULE['per_candidate_memory_bytes']):
            issues.append(f'{who}: memory.max={c["memory.max"]}, expected 536870912')
        if c['memory.swap.max']!='0':
            issues.append(f'{who}: memory.swap.max={c["memory.swap.max"]}, expected 0')
        if len(p['affinity'])<2:
            issues.append(f'{who}: fewer than two schedulable CPUs')
        if p['interfaces']!=['lo'] or container_settings[seat]['network']!='none':
            issues.append(f'{who}: network isolation unavailable')
        if p['python'][:2]!=[3,12] or p['numpy']!='1.26.4' or p['torch']!='2.5.1+cpu':
            issues.append(f'{who}: dependency versions {p["python"]}/{p["numpy"]}/{p["torch"]}')
        if p['helper_sha256']!=PLAN['files']['official_helpers']['sha256']:
            issues.append(f'{who}: official helper SHA differs or is missing')
    if probes[0]['cgroup_inode']==probes[1]['cgroup_inode']:
        issues.append('both containers see the same cgroup directory inode')
    return issues


def allowance(slow_steps, remainder):
    return min(RULE['step_hard_seconds'], 1 + remainder) if slow_steps<3 else 1


async def exchange(proc, payload=None, timeout=20):
    start = time.monotonic()
    async def request():
        if payload is not None:
            proc.stdin.write(encoded(payload))
            await proc.stdin.drain()
        line = await proc.stdout.readline()
        if not line:
            raise ValueError('worker stream closed')
        value = json.loads(line)
        if not isinstance(value,dict):
            raise ValueError('expected a JSON object')
        return value
    try:
        result = await asyncio.wait_for(request(), timeout)
        elapsed = time.monotonic()-start
        if elapsed>timeout:
            return dict(status='timeout', elapsed=elapsed, limit=timeout)
        return dict(status='ok', elapsed=elapsed, limit=timeout, response=result)
    except (TimeoutError, ValueError, OSError) as exc:
        return dict(status='timeout' if isinstance(exc,TimeoutError) else 'error', elapsed=time.monotonic()-start,
                    limit=timeout, error=type(exc).__name__ + ': ' +str(exc))


@asynccontextmanager
async def sandbox(image, role, cpus):
    name = 'sapling-live-'+uuid.uuid4().hex[:12]
    pack = ROOT/PLAN['files'][role]['path']
    args = ['docker','run','-i','--name',name,'--pull','never','--network','none',
            '--cpus','1.5','--memory','512m','--memory-swap','512m','--cpuset-cpus',cpus,'--read-only',
            '--tmpfs','/tmp:rw,nosuid,nodev,size=16m','--mount',f'type=bind,src={HERE/"worker.py"},dst=/worker.py,readonly',
            '--mount',f'type=bind,src={pack},dst=/submission.py,readonly',image,'python','-u','/worker.py',role]
    with tempfile.TemporaryFile() as log:
        proc = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                    stderr=log, limit=512*1024)
        try:
            probe = await exchange(proc, timeout=25)
            if probe['status']!='ok':
                log.seek(0)
                raise RuntimeError(f'container {role} preflight: {probe}, stderr={log.read(1000)!r}')
            inspected = json.loads(subprocess.check_output(['docker','inspect',name],timeout=10))[0]
            c = inspected['HostConfig']
            settings = dict(id=inspected['Id'], image=inspected['Image'], network=c['NetworkMode'],
                            nano_cpus=c['NanoCpus'], memory=c['Memory'], memory_swap=c['MemorySwap'],
                            cpuset=c['CpusetCpus'], read_only_root=c['ReadonlyRootfs'])
            yield proc, probe['response'], settings
        finally:
            cleanup = await asyncio.create_subprocess_exec('docker','rm','-f',name,stdout=asyncio.subprocess.DEVNULL,
                                                          stderr=asyncio.subprocess.DEVNULL)
            await cleanup.wait()
            if proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(),5)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()


def official():
    sys.path.insert(0, str(ROOT/'code/env_src/carbon_challenge-master'))
    import numpy as np
    from zerosum_env import make
    card_path = HERE.parent/'direct_original/games/candidate/raw_621758346_1.json.gz'
    if sha(card_path)!='7a5967383c78324d248f4b228541d7eb30f3ac3d39bf18e74f81a011a78c4631':
        raise ValueError('registered game configuration source differs')
    conf=json.loads(gzip.decompress(card_path.read_bytes()))['game_configuration']
    cfg={**conf,'agentTimeout':20,'actTimeout':1,'runTimeout':1500}
    np.random.seed(PLAN['seed'])
    env=make('carbon',configuration={**cfg,'randomSeed':PLAN['seed']})
    if hashlib.sha256(bytes(env.states[0].observation.full_carbon)).hexdigest()!=PLAN['physical_map_sha256']:
        raise ValueError('initial physical map mismatch')
    public_cfg={k:v for k,v in cfg.items() if k!='randomSeed'}
    return env,public_cfg


def observe(env,side,bank):
    shared=env.states[0].observation
    return dict(step=shared.step,player=side,carbon=env.states[side].observation.carbon,players=shared.players,
                trees=shared.trees,remainingOverageTime=bank)


def check_action(call,obs):
    if call['status']!='ok':
        return
    payload=call['response']; actions=payload.get('action')
    if payload.get('step')!=obs['step'] or not isinstance(actions,dict):
        call.update(status='error', error='invalid step/action payload'); return
    player=obs['players'][obs['player']]
    for uid,move in actions.items():
        legal=('UP','RIGHT','DOWN','LEFT') if uid in player[2] else ('RECCOLLECTOR','RECPLANTER') if uid in player[1] else ()
        if move not in legal:
            call.update(status='error', error=f'invalid command for {uid}'); return


def failures(calls):
    failed=[s for s,c in enumerate(calls) if c['status']!='ok']
    if len(failed)==2:
        return dict(status='bilateral_failure',counted=False,winner_seat=None)
    if failed:
        return dict(status='forfeit',counted=True,winner_seat=1-failed[0],failed_seat=failed[0],reason=calls[failed[0]]['status'])
    return None


def timing_by_seat(result, turns):
    # Keep the final timed-out exchange in the observed waits. Its wall value is
    # right-censored: the worker would have needed at least the allowed time.
    rows = [[], []]
    for turn in turns:
        for seat, call in enumerate(turn['calls']):
            rows[seat].append(call)
    timing = []
    for seat, calls in enumerate(rows):
        init = result.get('initialization', [{}, {}])[seat]
        waits = sorted(c['elapsed'] for c in calls)
        samples = [c.get('response', {}) for c in [init, *calls]]
        def quantile(p):
            return waits[math.ceil(p * len(waits)) - 1] if waits else None
        def peak(key):
            return max((s[key] for s in samples if s.get(key) is not None), default=None)
        timing.append(dict(seat=seat, role=ROLES[seat], logged_requests=len(calls),
                           init_seconds=init.get('elapsed'), init_status=init.get('status'),
                           timed_out_requests=sum(c['status']=='timeout' for c in calls),
                           error_requests=sum(c['status']=='error' for c in calls),
                           observed_waits_over_one_second=sum(w>1 for w in waits),
                           observed_excess_seconds=sum(max(0, w-1) for w in waits),
                           observed_wait_seconds_p95=quantile(.95), observed_wait_seconds_p99=quantile(.99),
                           observed_wait_seconds_max=quantile(1), reported_peak_rss_kib=peak('rss_peak_kib'),
                           reported_peak_cgroup_bytes=peak('cgroup_peak_bytes')))
    return timing


async def game(procs, result, output):
    env, cfg=official()
    ledger=output/'turns.jsonl'; bank=[5.,5.]; slow=[0,0]
    # Claim this registration before an actual attempt. An interrupted match cannot
    # be resampled by changing the output path or by calling this command twice.
    with CLAIM.open('x') as fd:
        fd.write(encoded({'seed':PLAN['seed'],'runtime_image':result['image_id'],'protocol_sha256':sha(REGISTRATION)}).decode())
    result['strategy_initializations']=2
    ready=await asyncio.gather(*(exchange(p,{'load':True},20) for p in procs))
    result['initialization']=ready
    for s,call in enumerate(ready):
        if call['status']=='ok':
            v=call['response']
            if not v.get('ready') or v.get('sha256')!=PLAN['files'][ROLES[s]]['sha256']:
                raise RuntimeError('cold load identity differs')
            if ROLES[s]=='opponent' and v.get('original')!={'_TEMP':1,'_SEED':None,'_S_ABORT':0.58,'_SEARCH_OK':True}:
                raise RuntimeError('original search/configuration unavailable')
    lost=failures(ready)
    if lost:return lost
    with ledger.open('x') as log:
        while not env.done:
            t=env.states[0].observation.step
            obs=[observe(env,s,bank[s]) for s in (0,1)]
            # Each turn both agents receive observations frozen before the move. Serial
            # exchange on the shared two allowed cores gives each agent its full quota.
            calls=[None,None]
            for s in (t%2,1-t%2):
                calls[s]=await exchange(procs[s],{'observation':obs[s],'configuration':cfg},allowance(slow[s],bank[s]))
                result['action_calls'][s]+=1
                check_action(calls[s],obs[s])
                if calls[s]['status']=='ok' and calls[s]['elapsed']>1:
                    slow[s]+=1; bank[s]=max(0.,bank[s]-(calls[s]['elapsed']-1.0))
            result['slow_steps']=slow.copy();result['overage_remaining_seconds']=bank.copy()
            log.write(encoded({'step':t,'calls':calls,'public_observation_sha256':[hashlib.sha256(encoded(o)).hexdigest() for o in obs],
                               'overage_remaining_seconds':bank,'slow_steps':slow}).decode());log.flush()
            lost=failures(calls)
            if lost:return lost
            env.step([c['response']['action'] for c in calls])
            result['official_transitions']=env.states[0].observation.step
    if [s.status for s in env.states]!=['DONE','DONE']:
        raise RuntimeError('unexpected official environment status')
    score=[s.reward for s in env.states]
    return dict(status='complete',counted=True,official_scores_by_seat=score,
                winner_seat=None if score[0]==score[1] else int(score[1]>score[0]))


async def run(args):
    for entry in PLAN['files'].values():
        path=ROOT/entry['path']
        if sha(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']:
            raise RuntimeError(f'registered file differs: {path}')
    if any(PLAN['files'][role]['bytes']>RULE['source_limit_bytes'] for role in ROLES):
        raise RuntimeError('submission file too large')
    dest=args.out; dest.mkdir(parents=True,exist_ok=True); receipt=dest/'receipt.json'
    if receipt.exists(): raise FileExistsError(receipt)
    result=dict(protocol_sha256=sha(REGISTRATION),runner_sha256=sha(Path(__file__)),worker_sha256=sha(HERE/'worker.py'),
                seed=PLAN['seed'],candidate_seat=ROLES.index('candidate'),official_transitions=0,action_calls=[0,0],strategy_initializations=0,
                docker_info=json.loads(subprocess.check_output(['docker','info','--format','{{json .}}']))['CgroupDriver'])
    result['image_id']=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    lock=None
    if args.play:
        lock=GAME_LOCK.open('a')
        fcntl.flock(lock,fcntl.LOCK_EX)
    start=time.monotonic()
    try:
        async with asyncio.timeout(RULE['task_seconds']):
            async with AsyncExitStack() as stack:
                sandboxes=[await stack.enter_async_context(sandbox(result['image_id'],role,args.cpus)) for role in ROLES]
                result['probes']=[x[1] for x in sandboxes]; result['containers']=[x[2] for x in sandboxes]
                issues=eligibility(result['probes'],result['containers']); result['blockers']=issues
                if issues:
                    result.update(status='BLOCKED_BEFORE_AGENT_LOAD',counted=False)
                elif args.play:
                    result.update(await game([x[0] for x in sandboxes],result,dest))
                else:
                    result.update(status='eligible_probe_only',counted=False)
    except TimeoutError:
        result.update(status='whole_task_timeout',counted=False)
    except Exception as e:
        result.update(status='infrastructure_error',counted=False,error=f'{type(e).__name__}: {e}')
    finally:
        if lock is not None:lock.close()
    result['whole_wall_seconds']=time.monotonic()-start
    if result['whole_wall_seconds']>RULE['task_seconds'] and result.get('status')=='complete':
        result.update(status='whole_task_timeout',counted=False,winner_seat=None)
    ledger=dest/'turns.jsonl'
    if ledger.is_file():
        result['turns_sha256']=sha(ledger)
        with ledger.open() as stream:
            result['timing_by_seat']=timing_by_seat(result, (json.loads(line) for line in stream))
    elif 'initialization' in result:
        result['timing_by_seat']=timing_by_seat(result, ())
    save(receipt,result)
    print(result['status'],result.get('blockers',[]),flush=True)
    return 0 if result['status'] in ('complete','forfeit','eligible_probe_only','bilateral_failure') else 2


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--image',required=True);cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--cpus',default='0,1');cli.add_argument('--play',action='store_true')
    sys.exit(asyncio.run(run(cli.parse_args())))
