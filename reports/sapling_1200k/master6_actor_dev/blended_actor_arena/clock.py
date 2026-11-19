"""Persist original examiner telemetry at the same commit boundary as the two physical commands."""
from collections import Counter
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import time
from unittest.mock import patch

import native_clock


SCHEMA = 'blended-original-clock/1'
FIRST = '0' * 64


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def chain(driver, previous, commands):
    return hashlib.sha256(bytes.fromhex(previous) + driver.pack(commands)).hexdigest()


def cpu_sample(host, cpu):
    """Outside the original agent. Host core ticks include processes in other cgroups."""
    prefix = f'cpu{cpu} '
    ticks = next(line[len(prefix):].split() for line in Path('/proc/stat').read_text().splitlines()
                 if line.startswith(prefix))
    root = Path('/sys/fs/cgroup')
    group = root / host['cgroup'].lstrip('/')
    if not (group / 'cpu.stat').is_file():
        group = root
    cgroup = dict(line.split() for line in (group / 'cpu.stat').read_text().splitlines())
    return dict(host_cpu_ticks=list(map(int, ticks[:8])), ticks_per_second=os.sysconf('SC_CLK_TCK'),
                cgroup_cpu={k:int(cgroup[k]) for k in ('usage_usec','nr_throttled','throttled_usec')},
                worker_cpu_seconds=time.process_time(), monotonic_seconds=time.perf_counter())


def check_tape(driver, clock, raw, identity):
    require(clock['schema']==SCHEMA and clock['identity']==identity, 'wrong clock job, CPU or identity')
    search = raw['search_calls'] if 'search_calls' in raw else raw['search']
    require(len(clock['calls'])==len(search)==len(raw['actions'])==clock['step'], 'incomplete clock action coverage')
    h = FIRST
    for t,(event,action,call) in enumerate(zip(search,raw['actions'],clock['calls'])):
        kind=call['kind']
        require(event==[t,kind] and call['step']==t and kind in native_clock.RETURNS,
                f'original search/clock mismatch at step {t}')
        require(all(native_clock.duration(call[k]) for k in ('public_wall_seconds','process_cpu_seconds')),
                'unusable public timing')
        d,k,roll,native=(call[x] for x in ('max_depth','candidates','rolls','native_seconds'))
        require(all(type(v) is int for v in (d,k,roll)) and 0 <= d <= min(8,299-t) and 1 <= k <=32,
                'invalid actual original search depth/candidates')
        require(type(call['native_aborted']) is bool and type(call['changed']) is bool,'invalid native counters')
        if kind=='single':
            require((k,d,roll,native,call['native_aborted'],call['changed'])==(1,0,0,None,False,False),
                    'invalid native one-candidate entry')
        else:
            require(k>=2 and roll==1 and native_clock.duration(native) and native<=call['public_wall_seconds']+1e-6,
                    'invalid searched timing or rollout')
            require(d>0 if kind=='searched' else (d==0 and call['native_aborted'] and not call['changed']),
                    'invalid original cutoff')
        h=chain(driver,h,action)
    require(h==clock['actions_chain_sha256'], 'clock bound to a different joint-action prefix')
    check_stats(clock['calls'],clock['native_stats'])
    require(clock['segments'] and clock['segments'][0]['first_step']==0, 'missing initial CPU sampling')
    start=0
    for segment in clock['segments']:
        require(segment['first_step']==start and segment['end_step']>start and segment['host']['affinity']==[identity['cpu']],
                'invalid clock process segment or CPU assignment')
        before,after=segment['start'],segment['end']
        require(after['monotonic_seconds']>=before['monotonic_seconds'] and after['worker_cpu_seconds']>=before['worker_cpu_seconds'],
                'invalid clock process sampling')
        start=segment['end_step']
    require(start==len(clock['calls']),'clock process segments do not cover actual calls')


def check_stats(calls, stats):
    completed=[v for v in calls if v['kind']=='searched']
    for field, recorded in (('max_depth','depth'),('candidates','cands'),('rolls','rolls')):
        actual=Counter(v[field] for v in completed)
        require(dict(actual)=={int(k):v for k,v in stats[recorded].items()}, 'original histogram disagrees with clock')
    require([v['native_seconds'] for v in completed]==stats['time'] and
            sum(v['native_aborted'] for v in calls)==stats['aborted'] and
            sum(v['changed'] for v in calls)==stats['changed'], 'original times or totals disagree with clock')


def verify_completed(driver, job):
    """Independently audit a previously completed wave without calling either agent."""
    arm,seed,seat,cpu,board,binding=job
    out=driver.OUT/arm
    wall=driver.read(out/f'wall_{seed}_{seat}.json.gz')
    rawfile=out/f'raw_{seed}_{seat}.json.gz'
    raw=driver.read(rawfile)
    signed=driver.read(out/f'receipt_{seed}_{seat}.json')
    row=signed['game']
    identity=dict(arm=arm,seed=seed,seat=seat,cpu=cpu,terrain=board,binding=binding)
    require(signed['binding']==binding and (raw['seed'],raw['candidate_seat'],raw['initial_terrain_u8_sha256'])==
            (seed,seat,board) and row['raw_sha256']==wall['raw_sha256']==driver.sha(rawfile),
            'source, seat, board or finalized clock identity changed')
    check_tape(driver,wall,raw,identity)
    if 'search' in row:
        check_stats(wall['calls'],row['search'])
    require(wall['summary']==native_clock.summarize(wall['calls'],raw['configuration']['_S_ABORT'],
            raw['configuration']['_S_D2']), 'clock summary differs from actual returns')
    return wall


def run_one(driver, job):
    arm,seed,seat,cpu,board,binding=job
    out=driver.OUT/arm
    def path(prefix,ext):
        return out/f'{prefix}_{seed}_{seat}.{ext}'
    rawfile=path('raw','json.gz')
    ck=path('progress','json.gz')
    mirror=path('wall_progress','json.gz')
    wall=path('wall','json.gz')
    receipt=path('receipt','json')
    identity=dict(arm=arm,seed=seed,seat=seat,cpu=cpu,terrain=board,binding=binding)
    def finalize(data, result):
        raw=driver.read(rawfile)
        require(raw['seed']==seed and raw['candidate_seat']==seat and raw['initial_terrain_u8_sha256']==board,
                'terminal replay differs from clock ticket')
        check_tape(driver,data,raw,identity)
        if 'search' in result['game']:
            check_stats(data['calls'],result['game']['search'])
        final={**data, 'raw_sha256':driver.sha(rawfile),
               'summary':native_clock.summarize(data['calls'],raw['configuration']['_S_ABORT'],raw['configuration']['_S_D2'])}
        require(result['game']['raw_sha256']==final['raw_sha256'],'official terminal receipt has different raw SHA')
        driver.agreed(wall,final)
        if mirror.exists():mirror.unlink()
        return result

    def checkpoint_data():
        if ck.exists():
            previous=driver.read(ck)
            require('_clock' in previous,'existing original progress has no clock; do not replay live decisions')
            saved=previous['_clock']
            check_tape(driver,saved,previous,identity)
            require(saved['state_sha256']==previous['state_sha256'],'checkpoint state differs from clock')
            check_stats(saved['calls'],previous['stats'])
            return saved
        if mirror.exists():
            require(rawfile.exists(),'clock without its authoritative action checkpoint; manual recovery required')
            return driver.read(mirror)['_clock']
        return None

    if wall.exists() or rawfile.exists() or receipt.exists():
        require(rawfile.exists(),'original terminal evidence missing; cannot replay an already measured game')
        if wall.exists():
            data=driver.read(wall)
            require(data['raw_sha256']==driver.sha(rawfile),'original terminal action SHA changed')
        else:
            data=checkpoint_data()
            require(data is not None,'completed original game has no signed clock evidence')
        check_tape(driver,data,driver.read(rawfile),identity)
        return finalize(data,driver.one(job))

    prior=checkpoint_data()
    segments=list(prior['segments']) if prior else []
    calls=list(prior['calls']) if prior else []
    rolling=[prior['actions_chain_sha256'] if prior else FIRST]
    records=[]
    loaded=driver.arena.prior.shared.module
    save=driver.save
    restore=driver.restore

    def load(name, alias):
        mod=loaded(name,alias)
        if alias=='master_original' and Path(name)==Path(driver.arena.prior.SOURCE):
            rec=native_clock.Recorder(mod)
            require(rec.start['affinity']==[cpu],'original examiner did not start on the assigned CPU')
            require(not records,'more than one original examiner instance')
            records.append(rec)
            begin=cpu_sample(rec.start,cpu)
            segments.append(dict(first_step=len(calls),end_step=len(calls),host=rec.start,start=begin,end=begin))
            mod.agent=rec.wrap(mod.agent)
        return mod

    def restore_and_sync(state,opp,sim):
        restore(state,opp,sim)
        require(len(records)==1 and state['step']==len(calls),'clock resume position differs')
        records[0].before=native_clock.counters(opp._S_STATS)

    def save_committed(p, value):
        if p!=ck:
            return save(p,value)
        require(len(records)==1 and not records[0].errors,'failed to capture original examiner call')
        rec=records[0]
        require(len(rec.calls)==1 and value['step']==len(calls)+1,'clock advanced twice or skipped a turn')
        call=rec.calls.pop()
        require(value['search'][-1]==[len(calls),call['kind']] and call['step']==len(calls),'actual call differs from original raw event')
        calls.append(call)
        rolling[0]=chain(driver,rolling[0],value['actions'][-1])
        segments[-1].update(end_step=len(calls),end=cpu_sample(rec.start,cpu))
        data=dict(schema=SCHEMA,identity=identity,step=len(calls),state_sha256=value['state_sha256'],
                  actions_chain_sha256=rolling[0],segments=segments,calls=calls,
                  native_stats={k:value['stats'][k] for k in ('depth','cands','rolls','time','aborted','changed')})
        # First commit the game and telemetry together. The mirror can only lag this transaction.
        save(ck,{**value,'_clock':data})
        # The mirror is a hard link to those same bytes, so each turn compresses only once.
        link=mirror.with_name(mirror.name+f'.{os.getpid()}.link')
        link.unlink(missing_ok=True)
        os.link(ck,link)
        os.replace(link,mirror)

    with ExitStack() as hooks:
        hooks.enter_context(patch.object(driver.arena.prior.shared,'module',load))
        hooks.enter_context(patch.object(driver,'restore',restore_and_sync))
        hooks.enter_context(patch.object(driver,'save',save_committed))
        result=driver.one(job)
    require(mirror.exists() and rawfile.exists(),'native runner did not publish a complete clock and action tape')
    return finalize(driver.read(mirror)['_clock'],result)
