#!/usr/bin/env python3
"""Persistent standard-int8 agent timing + bounded held-out behavior comparison.
Run from repo root with --small/--large NPZ paths and fresh --out/--cache dirs.
Timing is same-input, interleaved, single-core local diagnostic, not certification.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse, copy, hashlib, importlib.util, json, pathlib, platform, random, shutil, statistics, sys, time
import numpy as np

BEHAVIOR_SEEDS=list(range(620040000,620040004))
TIMING_SEEDS=list(range(620041000,620041004))
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def load_module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def dist_ns(vals):
    a=np.asarray(vals,dtype=np.float64)/1e6
    return {'n':len(vals),'mean_ms':float(a.mean()),'p50_ms':float(np.percentile(a,50)),
        'p95_ms':float(np.percentile(a,95)),'max_ms':float(a.max()),'p95_method':'numpy linear percentile'}

def prepare(weights,label,cache,packer):
    d=cache/label;d.mkdir(parents=True,exist_ok=True);w=d/'weights.npz';shutil.copyfile(weights,w);p=d/'agent.py'
    with np.load(w,allow_pickle=False) as z:
        meta=json.loads(str(z['meta']));arrays={k:z[k] for k in z.files if k!='meta'}
        if not all(np.isfinite(x).all() for x in arrays.values()):raise ValueError('Nonfinite weight')
        params=sum(x.size for x in arrays.values())
    packer.QMODE='int8';packer.main(str(w),str(p),teacher=2,shield=0,nocollide=0,endgame=0,pshield=0,rguard=0)
    return p,{'label':label,'weight_filename':pathlib.Path(weights).name,'weight_sha256':sha(weights),
        'numeric_tensor_count':len(arrays),'numeric_parameter_count':params,'cfg':meta['cfg'],
        'pack_bytes':p.stat().st_size,'pack_md5':hashlib.md5(p.read_bytes()).hexdigest(),'pack_sha256':sha(p)}

def observations(seeds,FastCarbon,opponent):
    # Neither candidate controls the corpus. The same reference-policy rollout
    # supplies both seat streams; each candidate keeps its own recurrent/history state.
    streams=[];corpus=hashlib.sha256()
    for seed in seeds:
        random.seed(seed);sim=FastCarbon().reset(seed=seed)
        ref=[load_module(opponent,f'reference_{seed}_{p}').make_agent() for p in (0,1)]
        seat_obs=[[],[]]
        for step in range(60):
            acts=[]
            for p in (0,1):
                o=sim.obs(p);seat_obs[p].append(copy.deepcopy(o));corpus.update(canonical({'seed':seed,'seat':p,'obs':o}))
                acts.append(ref[p](o,sim.config))
            sim.step(*acts)
            if any(not sim.player_workers[p] and not sim.player_trees[p] and sim.cash[p]<30 for p in (0,1)):
                raise RuntimeError('Reference corpus terminated early; no silent continuation')
        for seat in (0,1):streams.append((seed,seat,seat_obs[seat],dict(sim.config)))
    return streams,corpus.hexdigest()

def benchmark(packs,streams):
    samples=[];imports={k:[] for k in packs};times={k:{'cold':[],'warm':[]} for k in packs};actions_equal=0
    for n,(seed,seat,obs_stream,cfg) in enumerate(streams):
        agents={}
        for label in (['small','large'] if n%2==0 else ['large','small']):
            t=time.perf_counter_ns();agents[label]=load_module(packs[label],f'timing_{label}_{seed}_{seat}').agent
            imports[label].append(time.perf_counter_ns()-t)
        for step,obs in enumerate(obs_stream):
            # Counterbalance scheduling/cache order; never force one model always first.
            order=['small','large'] if (n+step)%2==0 else ['large','small'];dt={};acts={}
            expected=canonical(obs)
            for label in order:
                x=copy.deepcopy(obs);c=copy.deepcopy(cfg)
                t=time.perf_counter_ns();a=agents[label](x,c);dt[label]=time.perf_counter_ns()-t
                if canonical(x)!=expected or c!=cfg:raise RuntimeError('Agent mutated shared timing input')
                if not isinstance(a,dict):raise RuntimeError('Invalid agent response')
                acts[label]=a;times[label]['cold' if step==0 else 'warm'].append(dt[label])
            actions_equal+=acts['small']==acts['large']
            samples.append([seed,seat,step,len(obs['players'][seat][2]),len(obs['players'][1-seat][2]),order[0],dt['small'],dt['large']])
    summary={}
    for label in packs:
        summary[label]={'cold_first_agent_call':dist_ns(times[label]['cold']),'warm_agent_calls':dist_ns(times[label]['warm']),
            'module_load_shared_interpreter':dist_ns(imports[label]),'warm_calls_per_wall_second':1e9/statistics.fmean(times[label]['warm'])}
    summary['large_over_small_warm_mean']=summary['large']['warm_agent_calls']['mean_ms']/summary['small']['warm_agent_calls']['mean_ms']
    summary['warm_paired_ratios_mean']=statistics.fmean(r[7]/r[6] for r in samples if r[2]>0)
    summary['same_actions_count']=actions_equal;summary['same_actions_denominator']=len(samples)
    summary['unit_regimes']={}
    for lo,hi in [(0,2),(3,6),(7,50)]:
        ss=[r for r in samples if r[2]>0 and lo<=r[3]<=hi]
        if ss:summary['unit_regimes'][f'{lo}-{hi}']={label:dist_ns([r[6 if label=='small' else 7] for r in ss]) for label in packs}
    return {'summary':summary,'samples_columns':['seed','seat','state_t','own_workers','opp_workers','first_model','small_ns','large_ns'],'samples':samples}

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--small',type=pathlib.Path,required=True);a.add_argument('--large',type=pathlib.Path,required=True)
    a.add_argument('--cache',type=pathlib.Path,required=True);a.add_argument('--out',type=pathlib.Path,required=True)
    args=a.parse_args();repo=pathlib.Path(__file__).resolve().parents[2];code=repo/'code';sys.path.insert(0,str(code))
    if args.out.exists() and any(args.out.iterdir()):raise RuntimeError('Use fresh output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    if hasattr(os,'sched_getaffinity'):
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[0]})
    import cnet2_pack
    from fastsim import FastCarbon,_Worker
    smoke=load_module(repo/'reports/sapling_bootstrap/smoke_ab.py','sapling_shared_observer')
    checks=smoke.selftest(FastCarbon,_Worker)
    protocol={'version':1,'small':'551187-param sqrt-role-action BC baseline','large':'new architecture sqrt-role-action BC candidate',
      'teacher':2,'optional_guards':False,'quantization':'int8','steps':60,'behavior_seeds':BEHAVIOR_SEEDS,'timing_seeds':TIMING_SEEDS,
      'training_validation_seeds_excluded':list(range(620000000,620000006)),
      'timing':'Independent rule-vs-rule observations. Each model owns persistent history. Deepcopy outside timer. Model order alternates; first calls and module loads reported separately. No inference warmup omitted from accounting.',
      'behavior':'4 maps x both candidate seats, fixed policy_v0_5; t30/t60 observed, early terminal carried forward explicitly; no terminal-strength inference.',
      'source_sha256':{n:sha(code/n) for n in smoke.SOURCE_NAMES},'observer_sha256':sha(repo/'reports/sapling_bootstrap/smoke_ab.py'),
      'script_sha256':sha(__file__),'scope':'Local single-core diagnostic, not 1.5-core/512MiB platform resource certification or strength trial'}
    smoke.atomic_json(args.out/'protocol.json',protocol)
    packs={};identity={}
    for label,weights in [('small',args.small),('large',args.large)]:packs[label],identity[label]=prepare(weights,label,args.cache,cnet2_pack)
    t0=time.monotonic();cpu0=time.process_time();streams,corpus_sha=observations(TIMING_SEEDS,FastCarbon,code/'policy_v0_5.py')
    timing=benchmark(packs,streams);timing.update({'corpus_sha256':corpus_sha,'reference_games':4,'seat_streams':8,'paired_observations':480,
        'wall_seconds_including_corpus':time.monotonic()-t0,'cpu_seconds_including_corpus':time.process_time()-cpu0})
    smoke.atomic_json(args.out/'timing.json',timing)
    print(json.dumps({'phase':'timing_complete','summary':timing['summary']}),flush=True)
    behavior={k:[] for k in packs};t0=time.monotonic();cpu0=time.process_time()
    for n,seed in enumerate(BEHAVIOR_SEEDS):
        for side in (0,1):
            for label in (['small','large'] if (n+side)%2==0 else ['large','small']):
                r=smoke.play(seed,side,packs[label],code/'policy_v0_5.py',FastCarbon);events=r.pop('events')
                r['events_sha256']=hashlib.sha256(canonical(events)).hexdigest();r['team']={'candidate_seat':side,'opponent_seat':1-side}
                r['termination']='agent_error' if r['failed'] else ('early_extinction' if r['terminal'] else 'short_horizon_cap');behavior[label].append(r)
                if r['failed']:raise RuntimeError(f'Behavior error in {label}: {r["errors"]}')
                print(json.dumps({'arm':label,'seed':seed,'seat':side,'transitions':r['transitions'],'t60':r['snapshots'].get('60',{}).get('candidate',{})}),flush=True)
                smoke.atomic_json(args.out/'behavior.partial.json',behavior)
    behavior_result={label:{'games':sorted(rows,key=lambda x:(x['seed'],x['candidate_side'])),'summary':smoke.summarize(rows)} for label,rows in behavior.items()}
    smoke.atomic_json(args.out/'behavior.json',behavior_result);(args.out/'behavior.partial.json').unlink()
    env={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
         'thread_env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    try:
        from threadpoolctl import threadpool_info
        env['blas_backends']=[{k:v for k,v in row.items() if k!='filepath'} for row in threadpool_info()]
    except ImportError:env['blas_backends']='threadpoolctl unavailable; thread env set before importing numpy'
    result={'status':'complete','protocol':protocol,'identity':identity,'self_checks':checks,'environment':env,'timing_summary':timing['summary'],
       'behavior_summary':{k:v['summary'] for k,v in behavior_result.items()},
       'cost':{'timing_and_reference_cpu_seconds':timing['cpu_seconds_including_corpus'],'timing_and_reference_wall_seconds':timing['wall_seconds_including_corpus'],
               'behavior_cpu_seconds':time.process_time()-cpu0,'behavior_wall_seconds':time.monotonic()-t0,'excludes':'packaging/self-test; observer adds overhead to behavior only'},
       'artifacts':{n:sha(args.out/n) for n in ['protocol.json','timing.json','behavior.json']}}
    smoke.atomic_json(args.out/'comparison.json',result)
    print(json.dumps({'status':'complete','identity':identity,'timing':timing['summary']}),flush=True)
if __name__=='__main__':main()
