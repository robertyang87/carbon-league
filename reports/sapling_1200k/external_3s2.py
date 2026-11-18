#!/usr/bin/env python3
"""Eight full research games against original sampled/searching Master3s2.
NumPy + Torch environment required. This is NOT the registered 72-machine exam.
Example after the independent budget timing has explicitly finished:
  python reports/sapling_1200k/external_3s2.py --weights120 trained.npz \
    --timing-release-note 'budget agent confirmed timing finished at ... UTC'
The same maps/seats do not pair the opponent's _SEED=None internal randomness.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CARBON_DEBUG']='1'
import argparse, ast, concurrent.futures, datetime, hashlib, json, multiprocessing, pathlib, random, shutil, statistics, subprocess, sys, tempfile, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'reports/sapling_bootstrap'))
import transfer_3s2 as prior
from fastsim import FastCarbon
from eval_v53 import CFG, _extinct
SEEDS=[620050000,620050001]
LABELS=['sapling551k-sqrt','sapling1200k-sqrt']
EPISODE_STEPS=int(CFG['episodeSteps'])
SEARCH_KEYS=prior.SEARCH_KEYS+('_S_WINO','_S_ROLL','_S_RTEMP')

def small_stats(xs):
    return dict(n=len(xs),sum=sum(xs),mean=statistics.fmean(xs) if xs else None,min=min(xs) if xs else None,max=max(xs) if xs else None)

def play(job):
    label,pack,seed,side=job
    begun=prior.utc(); total0=time.monotonic(); cpu0=time.process_time()
    random.seed(7919*seed+104729*side+31)
    sim=FastCarbon(CFG).reset(seed=seed); observer=prior.Observer()
    candidate=prior.shared.module(pack,f'candidate_{label}_{seed}_{side}')
    opponent=prior.shared.module(prior.SOURCE,f'master3s2_{label}_{seed}_{side}')
    assert opponent._TEMP==1 and opponent._SEED is None and opponent._SEARCH_OK and opponent._RX
    assert candidate._OPTS==dict(teacher=2,shield=False,nocollide=False,endgame=False,pshield=False,rguard=False)
    config={k:getattr(opponent,k) for k in SEARCH_KEYS}
    from threadpoolctl import threadpool_info
    pools=threadpool_info()
    assert all(p['num_threads']==1 for p in pools if p['user_api']=='blas')
    agents=[None,None]; agents[side]=candidate.agent; agents[1-side]=opponent.agent
    times=[[],[]]; calls=[0,0]; errors=[0,0]; details=[]; actions_record=[]
    pre=sim.get_state(); extinct=[False,False]; snapshots={}; failed=False
    tree_income=[0.,0.]; cash_curve=[dict(state_t=0,cash_by_seat=list(sim.cash))]
    load_wall=time.monotonic()-total0; play0=time.monotonic()
    for step in range(1,EPISODE_STEPS):
        pre=sim.get_state(); acts=[]
        for p in (0,1):
            calls[p]+=1; t0=time.monotonic()
            try:
                act=agents[p](sim.obs(p),sim.config)
                if not isinstance(act,dict):raise TypeError('Non-dict action')
                ids=set(sim.player_workers[p])|{bid for bid,_ in sim.bases[p]}
                if set(act)-ids:raise ValueError('Foreign or unknown action entity')
                if any(v not in ('UP','RIGHT','DOWN','LEFT','RECCOLLECTOR','RECPLANTER',None) for v in act.values()):raise ValueError('Unknown action')
                acts.append(act)
            except Exception as e:
                errors[p]+=1;failed=True;acts.append({})
                details.append(dict(decision_t=step-1,seat=p,type=type(e).__name__,message=str(e)))
            finally:times[p].append(time.monotonic()-t0)
        if failed:break
        observer.step(sim,acts); actions_record.append(dict(decision_t=step-1,actions=acts))
        post=sim.get_state();extinct=[bool(_extinct(post,p)) for p in (0,1)]
        for tid,tree in post['trees'].items():tree_income[tree['player']]+=float(post['trees_dict'].get(tid,[None,0])[1])
        cash_curve.append(dict(state_t=sim.step_count,cash_by_seat=list(post['cash']),tree_income_cumulative_by_seat=list(tree_income)))
        if step in (60,100,200,299):
            snapshots[str(step)]={role:prior.snapshot(sim,observer,p,errors,calls) for role,p in [('candidate',side),('opponent',1-side)]}
            for role,p in [('candidate',side),('opponent',1-side)]:snapshots[str(step)][role]['cumulative_tree_absorption_income']=tree_income[p]
        if any(extinct):break
    elapsed=time.monotonic()-play0
    assert config=={k:getattr(opponent,k) for k in SEARCH_KEYS}
    post=sim.get_state(); full=not failed and (any(extinct) or sim.step_count>=EPISODE_STEPS-1)
    scores=([(sim.step_count-EPISODE_STEPS-1) if extinct[p] else (float(pre['cash'][p]) if any(extinct) else float(post['cash'][p])) for p in (0,1)] if full else None)
    margin=scores[side]-scores[1-side] if scores is not None else None
    s=opponent._S_STATS
    assert s['searched']>0 and s['searched']+s['single']==calls[1-side], 'Actual complete search path required'
    final={role:prior.snapshot(sim,observer,p,errors,calls) for role,p in [('candidate',side),('opponent',1-side)]}
    for role,p in [('candidate',side),('opponent',1-side)]:final[role]['cumulative_tree_absorption_income']=tree_income[p]
    return dict(arm=label,seed=seed,candidate_seat=side,started_at_utc=begun,finished_at_utc=prior.utc(),
        actual_transitions=sim.step_count,full_game=full,extinct_by_seat=extinct,
        termination='agent_error' if failed else ('extinction' if any(extinct) else 'official_episode_limit'),
        official_score_by_seat=scores,candidate_official_margin=margin,
        result=None if margin is None else ('win' if margin>0 else ('loss' if margin<0 else 'draw')),
        final_observed_cash_by_seat=list(post['cash']),pre_last_transition_cash_by_seat=list(pre['cash']),
        final_observed=final,snapshots=snapshots,cash_curve=cash_curve,agent_calls_by_seat=calls,agent_errors_by_seat=errors,errors=details,
        call_seconds_by_seat=[small_stats(x) for x in times],wall_seconds=time.monotonic()-total0,
        module_load_wall_seconds=load_wall,play_wall_seconds=elapsed,cpu_seconds=time.process_time()-cpu0,
        opponent_search_config=config,opponent_search_stats=s,opponent_search_ok=bool(opponent._SEARCH_OK),
        opponent_has_embedded_rollout_weights=bool(getattr(opponent,'_ROLL',None)),
        opponent_torch_threads=opponent._torch.get_num_threads(),threadpools=pools,
        final_state_sha256=prior.hashed(post),actions_sha256=prior.hashed(actions_record),
        accounting_events_sha256=prior.hashed(observer.events),
        positive_delivery_and_plant_events=[e for e in observer.events if e['event']=='plant' or (e['event']=='deposit' and e['amount']>0)])

def summarize(rows):
    out={}
    for label in LABELS:
        rr=[r for r in rows if r['arm']==label];valid=[r for r in rr if r['full_game']]
        out[label]=dict(scheduled_games=4,observed_games=len(rr),valid_complete_games=len(valid),
            wins=sum(r['result']=='win' for r in valid),losses=sum(r['result']=='loss' for r in valid),draws=sum(r['result']=='draw' for r in valid),
            official_margin=small_stats([r['candidate_official_margin'] for r in valid]),
            actual_transitions=small_stats([r['actual_transitions'] for r in rr]),
            early_extinction_games=sum(r['termination']=='extinction' for r in rr),
            candidate_errors=sum(r['agent_errors_by_seat'][r['candidate_seat']] for r in rr),
            opponent_errors=sum(r['agent_errors_by_seat'][1-r['candidate_seat']] for r in rr),
            game_wall_seconds=small_stats([r['wall_seconds'] for r in rr]),
            own_delivery_total=sum(r['final_observed']['candidate']['own_positive_delivery_carbon'] for r in rr),
            new_trees_total=sum(r['final_observed']['candidate']['new_trees_planted'] for r in rr))
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights551',type=pathlib.Path,default=ROOT/'data/weights/sapling_bootstrap/sqrt.npz')
    ap.add_argument('--weights120',type=pathlib.Path,default=pathlib.Path('/tmp/sapling120-run/trained.npz'))
    ap.add_argument('--out',type=pathlib.Path,default=ROOT/'reports/sapling_1200k/external_3s2.json')
    ap.add_argument('--timing-release-note',required=True)
    args=ap.parse_args()
    if args.out.exists():raise RuntimeError('Evidence exists; choose a fresh --out for a new stochastic run')
    if not all(p.is_file() for p in (args.weights551,args.weights120)):raise FileNotFoundError('Both frozen candidate weights required')
    import numpy,torch,cnet2_pack
    from threadpoolctl import threadpool_info
    assert prior.digest(prior.SOURCE,'md5')==prior.EXPECTED_MD5
    assert not any(20291101<=s<=20292100 for s in SEEDS)
    registration=ROOT/'reports/REGISTRATION_MASTER3S2_EXAMINER_20261116.md'
    assert '20291101–20292100' in registration.read_text()
    spec=ROOT/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json'
    sources=[pathlib.Path(__file__),pathlib.Path(prior.__file__),pathlib.Path(prior.shared.__file__),prior.SOURCE,registration,spec]+[ROOT/'code'/p for p in ['fastsim.py','eval_v53.py','cnet2_pack.py','cnet2_agent.py','cnet2_np.py','cnet2_feat.py']]
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in sources}
    old=json.loads((ROOT/'reports/sapling_bootstrap/transfer_3s2.json').read_text()); trans=sum(r['transitions'] for r in old['games'])
    unit=sum(r['wall_seconds'] for r in old['games'])/trans
    result=dict(schema='sapling1200k-master3s2-full-research-v1',status='running',started_at_utc=prior.utc(),
        repository_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=hashes,
        runtime=dict(python=sys.version,numpy=numpy.__version__,torch=torch.__version__,cpu_quota=(pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().strip() if pathlib.Path('/sys/fs/cgroup/cpu.max').exists() else None)),
        protocol=dict(seeds=SEEDS,candidate_seats=[0,1],games_per_model=4,process_workers=2,game_configuration=CFG,
            timing_release_note=args.timing_release_note,budget_timing_receipt_sha256=prior.digest(ROOT/'reports/sapling_1200k/runtime/timing.json'),opponent_source_md5=prior.EXPECTED_MD5,
            registered_wrapper_md5='b4cde262695d38270e16bde33875a443',wrapper_used=False,
            source_entrypoint='direct original source agent; independent research side',formal_72_exam=False,fixed_exam_seeds_used=False,
            candidate_quantization='official cnet2_pack int8',candidate_teacher=2,candidate_optional_guards=False,
            opponent_TEMP=1,opponent_SEED=None,paired_per_seed_inference=False,opponent_search_config_override=False,
            numpy_blas_threads=1,original_opponent_torch_threads=2,no_training_or_tuning=True,
            interpretation='Only four complete external diagnostic games per model on two new maps and both seats. Not a registered exam, promotion gate, population strength estimate, or seed-paired improvement estimate. Original _SEED=None and time-budget search preclude identical stochastic reruns.',
            terminal_scoring='eval_v53._extinct after each transition. Extinct score=t-episodeSteps-1; survivor of extinction gets pre-transition cash; normal episode limit gets final cash. Errors stop without fallback and invalidate outcome.',
            exception_mode='CARBON_DEBUG=1 surfaces outer agent/search exceptions. Actual search stats and network identity are also recorded; original internal network build fallbacks remain unchanged.'),
        prior_timing_estimate=dict(source='reports/sapling_bootstrap/transfer_3s2.json',source_sha256=prior.digest(ROOT/'reports/sapling_bootstrap/transfer_3s2.json'),total_transitions=trans,mean_game_wall_seconds_per_transition_including_import=unit,two_worker_ideal_8_by_299_seconds=unit*8*299/2,
            scope='Extrapolation from prior 60-step games, not a resource guarantee or an independent fresh timed benchmark.'),arms={},games=[])
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='sapling120-full-3s2-') as cache:
        packs={}
        for label,weight in zip(LABELS,[args.weights551,args.weights120]):
            local=pathlib.Path(cache)/(label+'.npz');shutil.copyfile(weight,local);pack=local.with_suffix('.py')
            cnet2_pack.QMODE='int8';cnet2_pack.main(str(local),str(pack),teacher=2,shield=0,nocollide=0,endgame=0,pshield=0,rguard=0)
            result['arms'][label]=dict(weight_filename=weight.name,weight_sha256=prior.digest(weight),weight_md5=prior.digest(weight,'md5'),pack_sha256=prior.digest(pack),pack_md5=prior.digest(pack,'md5'),pack_bytes=pack.stat().st_size)
            packs[label]=str(pack)
        prior.shared.atomic_json(args.out,result)
        jobs=[(label,packs[label],seed,side) for seed in SEEDS for side in (0,1) for label in LABELS]
        with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            futures=[pool.submit(play,j) for j in jobs]
            for future in concurrent.futures.as_completed(futures):
                row=future.result();result['games'].append(row);result['games'].sort(key=lambda r:(r['arm'],r['seed'],r['candidate_seat']))
                result['summary']=summarize(result['games']);prior.shared.atomic_json(args.out,result)
                print(json.dumps({k:row[k] for k in ('arm','seed','candidate_seat','actual_transitions','termination','candidate_official_margin','result','agent_errors_by_seat','wall_seconds')}),flush=True)
    for p in sources:assert hashes[str(p.relative_to(ROOT))]==prior.digest(p),f'Source changed while running: {p}'
    for label,weight in zip(LABELS,[args.weights551,args.weights120]):assert result['arms'][label]['weight_sha256']==prior.digest(weight)
    assert len(result['games'])==8 and all(r['full_game'] and not r['errors'] for r in result['games'])
    result['status']='complete';result['finished_at_utc']=prior.utc();result['elapsed_wall_seconds']=time.monotonic()-started
    prior.shared.atomic_json(args.out,result)
    print(json.dumps(dict(out=str(args.out),status=result['status'],elapsed_wall_seconds=result['elapsed_wall_seconds'])),flush=True)
if __name__=='__main__':main()
