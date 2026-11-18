#!/usr/bin/env python3
"""Research-side 60-transition behavior transfer check, NOT the registered exam.
Run in a NumPy + Torch environment from repository root:
python reports/sapling_bootstrap/transfer_3s2.py
Master3s2 keeps its sampling and complete original search configuration.
Its _SEED=None means reruns are procedural, not trajectory-identical or paired.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CARBON_DEBUG']='1'  # Surface existing agent/search exceptions, never silently replace an action.
import argparse, collections, concurrent.futures, datetime, hashlib, importlib.util, inspect, json, multiprocessing, pathlib, random, shutil, statistics, subprocess, sys, tempfile, time

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import smoke_ab as shared
from fastsim import FastCarbon, _Worker
from eval_v53 import CFG, _extinct
SOURCE=ROOT/'data/master3s2_source.py'
EXPECTED_MD5='57f9a594cce4f2ec9e3573d8d9d26ca1'
SEEDS=[620030000,620030001]
SEARCH_KEYS=('_S_ENGINE','_S_START','_S_K','_S_GEN','_S_DUPS','_S_M','_S_MMAX','_S_STAGES','_S_D2','_S_FILL','_S_ROWS','_S_EPS','_S_ABORT','_S_SAFE','_S_CFIX','_S_HOMEFIX','_RETIRE','_BANK','_HOME','_TEMP','_SEED','_SID')

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p,algo='sha256'):return hashlib.new(algo,pathlib.Path(p).read_bytes()).hexdigest()
def hashed(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class Observer(shared.Observer):
    def __init__(self):
        super().__init__(FastCarbon)
        lines,start=inspect.getsourcelines(FastCarbon.step)
        hits=[start+i for i,s in enumerate(lines) if s.strip()=='self.cash[w.player] -= player_plant_cost[w.player]']
        if len(hits)!=1:raise RuntimeError('Plant accounting source changed')
        self.plant_line=hits[0]
    def trace(self,frame,event,arg):
        if frame.f_code is self.code and event=='line' and frame.f_lineno==self.plant_line:
            x=frame.f_locals; w=x['w']; amount=float(x['player_plant_cost'][w.player])
            self.d[w.player]['plant_spend']+=amount
            self.events.append(dict(step=x['self'].step_count+1,player=w.player,worker=w.id,event='plant',position=w.pos,amount=amount))
            return self.trace
        return super().trace(frame,event,arg)

def snapshot(sim,obs,p,errors,calls):
    out=shared.snapshot(sim,obs,p,errors,calls)
    out['plant_spend']=float(obs.d[p]['plant_spend'])
    return out

def one_game(job):
    label,pack,seed,side=job
    begun=utc(); all0=time.monotonic(); cpu0=time.process_time()
    random.seed(7919*seed+104729*side+31)
    sim=FastCarbon(CFG).reset(seed=seed); mon=Observer()
    a=shared.module(pack,f'candidate_{label}_{seed}_{side}')
    b=shared.module(SOURCE,f'master3s2_{label}_{seed}_{side}')
    assert b._TEMP==1.0 and b._SEED is None and b._SEARCH_OK and b._RX, 'Complete sampled search opponent required'
    assert a._OPTS==dict(teacher=2,shield=False,nocollide=False,endgame=False,pshield=False,rguard=False)
    frozen={k:getattr(b,k) for k in SEARCH_KEYS}
    # Do not override source's torch.set_num_threads(2). NumPy BLAS is one thread;
    # the original opponent's explicit Torch two-thread setting is separately recorded.
    try:
        from threadpoolctl import threadpool_info
        pools=threadpool_info()
        assert all(p['num_threads']==1 for p in pools if p['user_api']=='blas'),pools
    except ImportError:pools=None
    load_wall=time.monotonic()-all0
    agents=[None,None]; agents[side]=a.agent; agents[1-side]=b.agent
    errors=[0,0]; calls=[0,0]; error_records=[]; snaps={}; action_trace=[]; extinct=[False,False]
    durations=[[],[]]; play0=time.monotonic(); pre=sim.get_state(); failed=False
    for transition in range(1,61):
        pre=sim.get_state(); acts=[]
        for p in (0,1):
            t0=time.monotonic(); calls[p]+=1
            try:
                act=agents[p](sim.obs(p),sim.config)
                if not isinstance(act,dict):raise TypeError('Non-dict action')
                ids=set(sim.player_workers[p])|{bid for bid,_ in sim.bases[p]}
                if set(act)-ids:raise ValueError('Foreign or unknown action entity')
                if any(v not in ('UP','RIGHT','DOWN','LEFT','RECCOLLECTOR','RECPLANTER',None) for v in act.values()):raise ValueError('Unknown action')
                acts.append(act)
            except Exception as e:
                errors[p]+=1; failed=True
                error_records.append(dict(transition=transition,seat=p,type=type(e).__name__,message=str(e)))
                acts.append({})
            finally:durations[p].append(time.monotonic()-t0)
        if failed:break  # No fabricated fallback transition.
        mon.step(sim,acts)
        action_trace.append(dict(decision_t=transition-1,actions=acts))
        post=sim.get_state(); extinct=[bool(_extinct(post,p)) for p in (0,1)]
        if transition in (30,60):snaps[str(transition)]={role:snapshot(sim,mon,p,errors,calls) for role,p in [('candidate',side),('opponent',1-side)]}
        if any(extinct):break
    assert frozen=={k:getattr(b,k) for k in SEARCH_KEYS},'Search configuration changed during test'
    final={role:snapshot(sim,mon,p,errors,calls) for role,p in [('candidate',side),('opponent',1-side)]}
    final_state=sim.get_state()
    def timing(xs):return dict(calls=len(xs),total=sum(xs),mean=statistics.fmean(xs) if xs else None,max=max(xs) if xs else None)
    return dict(arm=label,seed=seed,candidate_seat=side,started_at_utc=begun,finished_at_utc=utc(),
        transitions=sim.step_count,early_terminal=any(extinct),extinct_by_seat=extinct,
        termination='agent_error' if failed else ('extinction' if any(extinct) else '60_transition_behavior_cap'),
        agent_calls_by_seat=calls,agent_errors_by_seat=errors,errors=error_records,snapshots=snaps,final_observed=final,
        terminal_score_by_seat=([(sim.step_count-301) if extinct[p] else float(pre['cash'][p]) for p in (0,1)] if any(extinct) else None),
        scores_not_defined_at_behavior_cap=not any(extinct),wall_seconds=time.monotonic()-all0,
        play_wall_seconds=time.monotonic()-play0,module_load_wall_seconds=load_wall,cpu_seconds=time.process_time()-cpu0,
        call_timing_by_seat=[timing(x) for x in durations],opponent_search_config=frozen,
        opponent_search_ok=bool(b._SEARCH_OK),opponent_embedded_engine=bool(b._RX),opponent_search_stats=b._S_STATS,
        opponent_torch_threads=b._torch.get_num_threads(),threadpools=pools,
        final_state_sha256=hashed(final_state),action_trace_sha256=hashed(action_trace),
        measured_events_sha256=hashed(mon.events),
        positive_delivery_and_plant_events=[e for e in mon.events if e['event']=='plant' or (e['event']=='deposit' and e['amount']>0)])

def summary(rows):
    out={}
    for label in ('initial','sqrt-role-action'):
        rr=[r for r in rows if r['arm']==label]; z=dict(games=len(rr),transitions=sum(r['transitions'] for r in rr),early_terminal_games=sum(r['early_terminal'] for r in rr),wall_seconds_sum=sum(r['wall_seconds'] for r in rr))
        for role in ('candidate','opponent'):
            vals=[r['final_observed'][role] for r in rr]
            fields=('positive_delivery_events','own_positive_delivery_carbon','closed_loop_events','new_trees_planted','plant_spend','cash','cargo','alive_workers','alive_trees')
            z[role]=dict(games_with_positive_delivery=sum(v['positive_delivery_events']>0 for v in vals),games_with_new_trees=sum(v['new_trees_planted']>0 for v in vals),games_with_completed_collection_delivery_loop=sum(v['completed_collection_delivery_loop'] for v in vals),agent_errors=sum(v['agent_errors'] for v in vals),
                totals={k:sum(v[k] for v in vals) for k in fields},means={k:statistics.fmean(v[k] for v in vals) if vals else None for k in fields})
        out[label]=z
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=pathlib.Path,default=ROOT/'reports/sapling_bootstrap/transfer_3s2.json')
    ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    if args.out.exists():raise RuntimeError('Evidence already exists; pass a fresh --out for reruns')
    if args.workers not in (1,2):raise ValueError('At most two workers: each original opponent sets two Torch threads')
    assert digest(SOURCE,'md5')==EXPECTED_MD5
    reg=ROOT/'reports/REGISTRATION_MASTER3S2_EXAMINER_20261116.md'
    assert '20291101–20292100' in reg.read_text() and not any(20291101<=s<=20292100 for s in SEEDS)
    import torch, numpy
    checks=shared.selftest(FastCarbon,_Worker)
    sources=[pathlib.Path(__file__),pathlib.Path(shared.__file__),SOURCE,reg]+[ROOT/'code'/n for n in ('cnet2_pack.py','cnet2_np.py','cnet2_agent.py','cnet2_feat.py','fastsim.py','eval_v53.py')]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
    result=dict(schema='sapling-transfer-3s2-research-v1',status='running',started_at_utc=utc(),
        runtime=dict(python=sys.version,python_executable=sys.executable,numpy=numpy.__version__,torch=torch.__version__),
        initial_preflight_note='The first launch in the default Python had no Torch; full-search assertion rejected before any transition. This completed run uses an installed Torch environment, with _SEARCH_OK asserted per game.',
        repository_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=hashes,
        protocol=dict(seeds=SEEDS,candidate_seats=[0,1],games_per_arm=4,max_transitions=60,
            opponent_source=str(SOURCE.relative_to(ROOT)),opponent_source_md5=EXPECTED_MD5,
            registered_wrapper_md5='b4cde262695d38270e16bde33875a443',wrapper_used=False,
            entrypoint='direct original source agent; research side',formal_72_reexamination=False,
            registration_review='Sections 2/5 define registered result/exam use and fixed maps. This separate short behavior study uses neither fixed maps nor formal exam verdicts; no cash handicap.',
            quantization='official cnet2_pack int8',teacher=2,all_candidate_optional_guards=False,
            opponent_TEMP=1.0,opponent_SEED=None,paired_gain_inference=False,
            interpretation='Only early behavior transfer on two new maps and both seats. Internal opponent RNG is fresh per game; time-budgeted search also depends on hardware. No terminal strength, winrate, resource certification, or paired-gain inference.',
            source_error_mode='CARBON_DEBUG=1 exposes original search/agent exceptions; any exception stops game without fallback continuation.',
            blas_threads=1,original_opponent_torch_threads=2,process_workers=args.workers,
            no_training_or_parameter_tuning=True,registration_source_sha256=digest(reg)),observer_selftests=checks,arms={},games=[])
    all0=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='sapling-transfer-3s2-') as tmp:
        import cnet2_pack
        jobs=[]
        for label,name in [('initial','initial.npz'),('sqrt-role-action','sqrt.npz')]:
            weight=ROOT/'data/weights/sapling_bootstrap'/name
            local=pathlib.Path(tmp)/(label+'.npz'); shutil.copyfile(weight,local); pack=pathlib.Path(tmp)/(label+'.py')
            cnet2_pack.QMODE='int8'; cnet2_pack.main(str(local),str(pack),teacher=2,shield=0,nocollide=0,endgame=0,pshield=0,rguard=0)
            result['arms'][label]=dict(weights=str(weight.relative_to(ROOT)),weight_sha256=digest(weight),weight_md5=digest(weight,'md5'),pack_sha256=digest(pack),pack_md5=digest(pack,'md5'),pack_bytes=pack.stat().st_size)
            jobs.extend((label,str(pack),seed,side) for seed in SEEDS for side in (0,1))
        shared.atomic_json(args.out,result)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            futures={pool.submit(one_game,j):j for j in jobs}
            for f in concurrent.futures.as_completed(futures):
                row=f.result();result['games'].append(row);result['games'].sort(key=lambda r:(r['arm'],r['seed'],r['candidate_seat']))
                result['summary']=summary(result['games']);shared.atomic_json(args.out,result)
                v=row['final_observed']['candidate']
                print(json.dumps(dict(arm=row['arm'],seed=row['seed'],side=row['candidate_seat'],transitions=row['transitions'],terminal=row['early_terminal'],deliveries=v['positive_delivery_events'],delivery_amount=v['own_positive_delivery_carbon'],planted=v['new_trees_planted'],errors=row['agent_errors_by_seat'],wall_seconds=row['wall_seconds'])),flush=True)
    if len(result['games'])!=8 or any(r['errors'] for r in result['games']):raise RuntimeError('Incomplete or failed research: diagnostic evidence retained')
    for p in sources:
        assert hashes[str(p.relative_to(ROOT))]==digest(p),f'Source changed while running: {p}'
    result['status']='complete';result['finished_at_utc']=utc();result['elapsed_wall_seconds']=time.monotonic()-all0;shared.atomic_json(args.out,result)
    print(json.dumps(dict(out=str(args.out),status=result['status'],elapsed_wall_seconds=result['elapsed_wall_seconds'])),flush=True)
if __name__=='__main__':main()
