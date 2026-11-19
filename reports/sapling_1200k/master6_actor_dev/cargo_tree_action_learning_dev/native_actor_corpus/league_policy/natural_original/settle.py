"""Reconcile all three real policies per map and seat from the signed hosted request/response tapes."""
import argparse
import ast
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parents[2]
sys.path.insert(0, str(FAMILY/'territory_sequence/fresh_original'))
from verify_pair import verify_game, live

ARMS = ('greedy_parent','sampled_parent','sampled_child')
DIGESTS = dict(greedy_parent='8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac',
               sampled_parent='93d70be43a8cc358c05e4ac3d0870bded1e720aa539208ed0550523fc718e789',
               sampled_child='851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee',
               opponent='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5')
ARMDIR = dict(sampled_parent='parent',sampled_child='candidate')


def require(condition, why):
    if not condition:
        raise ValueError(why)


def read(path):
    return json.loads(path.read_bytes())


def freeze_inputs(matrix_file):
    m = read(matrix_file)
    require(len(m['seeds']) == len(set(m['seeds'])) == len(set(m['physical_boards'].values())) == 2, 'Wrong map cohort')
    for arm,sha in DIGESTS.items():
        entry = m['files'][arm]
        require(entry['sha256'] == sha, f'Wrong fixed package for {arm}')
        file = live.ROOT/entry['path']
        require(live.sha(file) == sha, f'Wrong local bytes for {arm}')
        if arm in ARMDIR:
            # Bind both the runtime RNG expression and the proof from repeated draws at one
            # fixed, already exposed public observation. A second cold process has its own draw.
            group = ARMDIR[arm]
            probe = read(HERE/(group+'_natural.json'))
            a = read(HERE/(group+'_audit.json'))
            require(probe['package_sha256'] == a['deployment']['sha256'] == sha, 'Stale RNG evidence')
            rows = probe['training_roots']
            require({z['seat'] for z in rows} == {0,1} and len(rows)==2
                    and len({z['first_rng_state_sha256'] for z in rows})==2
                    and all(min(z['counts'])>0 and sum(z['counts'])==z['trials'] for z in rows), 'No two-choice public support')
            source = file.read_text()
            tree = ast.parse(source)
            final_entry = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='agent'][-1]
            rng = [n for n in ast.walk(final_entry) if isinstance(n,ast.Assign)
                   and any(isinstance(v,ast.Name) and v.id=='_behavior_rng' for v in n.targets)]
            require(len(rng)==1 and ast.unparse(rng[0].value) ==
                    "np.random.default_rng(int.from_bytes(os.urandom(16), 'little'))", 'Draw source differs')
            require('SAPLING_NATIVE_STREAM' not in source and 'SAPLING_NATIVE_TRACE' not in source,
                    'Training-only controls in release')
    return m


def check_timing(directory, verified):
    receipt = read(directory/'receipt.json')
    eligible = verified['status'] in ('complete','forfeit','bilateral_failure')
    if eligible:
        containers, probes = receipt['containers'],receipt['probes']
        require(len(containers)==len(probes)==2 and len({c['id'] for c in containers})==2,
                'Missing independent containers')
        require(not live.eligibility(probes,containers),'Cgroup or Python configuration differs')
        for c in containers:
            require(c['network']=='none' and c['read_only_root'] and c['nano_cpus']==1_500_000_000
                    and c['memory']==c['memory_swap']==536870912, 'External resource limits differ')
    whole = receipt['whole_wall_seconds']
    require(math.isfinite(whole) and whole>=0,'Invalid full match clock')
    if verified['status']=='complete':
        require(whole<=1500, 'Completed game passed beyond full-match time cap')
    clocks=verified['timing_by_seat'] or []
    if verified['status']=='complete':
        require(len(clocks)==2 and len(receipt['initialization'])==2,'Missing public request clocks')
        for stats in clocks:
            require(stats['observed_waits_over_one_second']<=3 and stats['observed_excess_seconds']<=5.00000001
                    and stats['observed_wait_seconds_max']<=3 and stats['reported_peak_cgroup_bytes'] is not None,
                    'Completed game violates limits or misses cgroup high-water mark')
    raw = directory/'turns.jsonl'
    totals = Counter()
    depth, width = Counter(), Counter()
    elapsed, cpu = [], []
    prior = Counter()
    if raw.is_file():
        with raw.open() as stream:
            for line in stream:
                t=json.loads(line)
                call=t['calls'][1-verified['seat']]
                if call['status']!='ok':
                    totals[call['status']]+=1
                    continue
                v=call['response']
                dt,pt=v['public_wall_seconds'],v['public_cpu_seconds']
                require(all(math.isfinite(w) and w>=0 for w in (dt,pt)), 'Invalid examiner clock')
                elapsed.append(dt); cpu.append(pt)
                current=Counter(v['search_counts'])
                require(set(current)=={'single','searched','aborted','changed'} and
                        all(0<=current[k]-prior[k]<=1 for k in current),'Invalid native search counter increments')
                change=current-prior
                prior=current
                last=v.get('search_last')
                if change['single']:
                    require(not change['searched'] and last and last['candidates']==1,'Single candidate counter mismatch')
                    kind='single_candidate'
                elif change['searched']:
                    require(last and last['depth']>0 and last['candidates']>1,'Missing completed search round')
                    kind='positive_depth'
                elif change['aborted']:
                    require(last and last['depth']==0 and last['candidates']>1,'Incorrect depth-zero fallback')
                    kind='zero_depth'
                else:
                    # In particular an unsearched call must not inherit a previous call's 'last'.
                    kind='unclassified'
                totals[kind]+=1
                if kind!='unclassified':
                    depth[int(last['depth'])]+=1
                    width[int(last['candidates'])]+=1
    return dict(examiner_call_classes=dict(totals),candidate_count_histogram=dict(width),
                completed_depth_histogram=dict(depth),examiner_successful_public_calls=len(elapsed),
                examiner_cpu_seconds=sum(cpu),examiner_wall_seconds=sum(elapsed),
                executor_verified_for_scored_attempt=eligible)


def verify_triple(directory,seed,seat,plan_path):
    directory,plan_path=Path(directory),Path(plan_path)
    m=freeze_inputs(plan_path)
    require(seed in m['seeds'] and seat in (0,1),'Map/seat outside development registration')
    games,errors={},{}
    for arm in ARMS:
        arm_dir=directory/arm
        try:
            verified=verify_game(arm_dir,seed,seat,arm,plan_path)
            receipt=read(arm_dir/'receipt.json')
            if (arm_dir/'attempt.json').is_file():
                claim=read(arm_dir/'attempt.json')
                require(claim==dict(seed=seed,runtime_image=receipt['image_id'],
                                    protocol_sha256=live.sha(arm_dir/'registration.json')),'Attempt identity')
            elif receipt.get('strategy_initializations',0)>0:
                raise ValueError('Started agents without a unique attempt ticket')
            verified['timing_audit']=check_timing(arm_dir,verified)
            games[arm]=verified
        except (OSError,ValueError,TypeError,KeyError,IndexError,AssertionError) as e:
            errors[arm]=f'{type(e).__name__}: {e}'
    for k in ('image_id','workflow_run'):
        if len({v[k] for v in games.values()})>1:
            errors['common_'+k]='Different '+k+' within the map/seat triple'
    return dict(seed=seed,seat=seat,games=games,verification_errors=errors)


def gh(tail):
    return json.loads(subprocess.check_output(['gh','api','repos/robertyang87/carbon-league/actions/'+tail]))


def summarize(games):
    summary={}
    for arm in ARMS:
        rows=[g for g in games if g['arm']==arm]
        subsets={'all':rows,'seat0':[g for g in rows if g['seat']==0],'seat1':[g for g in rows if g['seat']==1]}
        summary[arm]={}
        for name,cohort in subsets.items():
            wins=losses=draws=0
            margins=[]
            for g in cohort:
                if g['counted']:
                    if g['winner_seat'] is None:
                        draws+=int(g['status']=='complete')
                    elif g['winner_seat']==g['seat']:
                        wins+=1
                    else:
                        losses+=1
                score=g['official_scores_by_seat']
                if score is not None:
                    margins.append(score[g['seat']]-score[1-g['seat']])
            n=4 if name=='all' else 2
            summary[arm][name]=dict(planned=n,verified=len(cohort),wins=wins,draws=draws,losses=losses,
                undecided_in_planned=n-wins-draws-losses,official_numeric_scores=len(margins),
                mean_official_margin_all=statistics.mean(margins) if len(margins)==n else None,
                mean_completed_margin=statistics.mean(margins) if margins else None,
                status_counts=dict(Counter(g['status'] for g in cohort)))
    return summary


def contrasts(games,seeds):
    completed={(g['seed'],g['seat'],g['arm']):g for g in games if g['status']=='complete'}
    pairs={}
    for trial,control in [('sampled_child','sampled_parent'),('sampled_parent','greedy_parent'),('sampled_child','greedy_parent')]:
        paired=[]
        for seed in seeds:
            for side in (0,1):
                a=completed.get((seed,side,trial))
                b=completed.get((seed,side,control))
                if not a or not b:
                    continue
                def margin(g):
                    scores=g['official_scores_by_seat']
                    return scores[side]-scores[1-side]
                paired.append(dict(seed=seed,seat=side,official_margin_delta=margin(a)-margin(b)))
        by_map={s:statistics.mean(p['official_margin_delta'] for p in paired if p['seed']==s)
                for s in seeds if sum(p['seed']==s for p in paired)==2}
        pairs[trial+' minus '+control]=dict(paired_numeric_games=len(paired),pairs=paired,complete_map_means=by_map,
            mean_over_fixed_maps=statistics.mean(by_map.values()) if len(by_map)==len(seeds) else None)
    return pairs


def merge(args):
    run_id,commit=args.run,args.commit
    run=gh(f'runs/{run_id}')
    require(run['id']==run_id and run['head_sha']==commit and run['event']=='workflow_dispatch'
            and run['head_branch']=='main' and run['run_attempt']==1 and run['status']=='completed','Wrong GitHub run')
    plan_path=args.plan.resolve()
    matrix=freeze_inputs(plan_path)
    signed=subprocess.check_output(['git','-C',str(live.ROOT),'show',commit+':'+str(plan_path.relative_to(live.ROOT))])
    require(signed==plan_path.read_bytes(),'Local plan differs from GitHub run commit')
    inventory=gh(f'runs/{run_id}/artifacts?per_page=100')
    expected={(s,side) for s in matrix['seeds'] for side in (0,1)}
    remote={r['digest']:r for r in inventory['artifacts']}
    require(len(remote)==len(inventory['artifacts'])==inventory['total_count'],'Ambiguous GitHub archive digest')
    target=args.out.resolve()
    target.mkdir(parents=True,exist_ok=True)
    seen=set()
    report=dict(run_id=run_id,commit=commit,url=run['html_url'],run_conclusion=run['conclusion'],
                map_scope='Two previously exposed physical development maps. These matches do not enter the independent exam.',
                plan_sha256=live.sha(plan_path),archives=[],games=[],verification_errors=[])
    sys.path.insert(0,str(HERE))
    from audit_menu import verify_menu
    for path in args.archives:
        sha=live.sha(path)
        item=remote.pop('sha256:'+sha,None)
        require(item is not None and item['size_in_bytes']==path.stat().st_size
                and item['workflow_run']['head_sha']==commit and item['workflow_run']['id']==run_id,'Archive is not an authentic artifact of this run')
        attachment=dict(id=item['id'],name=item['name'],sha256=sha,bytes=path.stat().st_size,expired=item['expired'])
        report['archives'].append(attachment)
        # Keep even malformed or interrupted raw artifacts as evidence.
        dest=target/(str(item['id'])+'.zip')
        if dest.exists():
            require(live.sha(dest)==sha,'Destination contains different evidence')
        else:
            shutil.copyfile(path,dest)
        try:
            with zipfile.ZipFile(path) as z, TemporaryDirectory() as tmp:
                members=z.namelist()
                require(len(set(members))==len(members) and z.testzip() is None,'Damaged archive')
                pair=json.loads(z.read('pair_status.json'))
                key=(pair['seed'],pair['seat'])
                require(key in expected and key not in seen and str(pair['run_id'])==str(run_id)
                        and pair['commit']==commit and pair['protocol_sha256']==live.sha(plan_path),'Wrong hosted ticket')
                require(item['name']==f'natural-original-{key[0]}-{key[1]}-{run_id}','Archive label differs from the contained ticket')
                seen.add(key)
                attachment.update(seed=key[0],seat=key[1],job_outcome=pair.get('pair_step_outcome'))
                local=Path(tmp)
                for arm in ARMS:
                    (local/arm).mkdir()
                    for file in ('registration.json','receipt.json','attempt.json','turns.jsonl'):
                        name=arm+'/'+file
                        if name in members:
                            (local/name).write_bytes(z.read(name))
                checked=verify_triple(local,*key,plan_path)
                hosted=json.loads(z.read('verification.json')) if 'verification.json' in members else None
                if hosted!=json.loads(json.dumps(checked)):
                    report['verification_errors'].append(dict(seed=key[0],seat=key[1],error='Hosted vs local audit mismatch or missing hosted verification'))
                for arm,why in checked['verification_errors'].items():
                    report['verification_errors'].append(dict(seed=key[0],seat=key[1],arm=arm,error=why))
                for arm,g in checked['games'].items():
                    require(str(g['workflow_run'])==str(run_id),'Arm from a different workflow run')
                    report['games'].append(g)
                    if g['status'] in ('complete','forfeit','bilateral_failure','whole_task_timeout','infrastructure_error'):
                        try:
                            proof=verify_menu(local/arm,ARMDIR.get(arm,arm),persist=False)
                            require((g['status']!='complete' or proof['matched_all_steps']) and proof['deployment_sha256']==DIGESTS[arm],
                                    'Unsampled or illegal issued action')
                            g['action_support']=proof
                        except (OSError,ValueError,TypeError,KeyError,IndexError,AssertionError) as e:
                            report['verification_errors'].append(dict(seed=key[0],seat=key[1],arm=arm,error=str(e)))
        except (OSError,ValueError,TypeError,KeyError,IndexError,AssertionError,zipfile.BadZipFile) as e:
            report['verification_errors'].append(dict(archive=item['id'],error=f'{type(e).__name__}: {e}'))
    report['remote_artifacts_missing_locally']=[r['id'] for r in remote.values()]
    report['missing_triples']=sorted(expected-seen)
    report['summary']=summarize(report['games'])
    report['descriptive_contrasts']=contrasts(report['games'],matrix['seeds'])
    report['records_reconciled']=(len(report['games'])==12 and not report['verification_errors']
                                  and not report['missing_triples'] and not remote)
    output=target/'settlement.json'
    live.save(output,report)
    return report['records_reconciled']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='mode',required=True)
    verify=commands.add_parser('verify')
    verify.add_argument('directory',type=Path)
    verify.add_argument('--seed',required=True,type=int)
    verify.add_argument('--seat',required=True,choices=(0,1),type=int)
    merger=commands.add_parser('merge')
    merger.add_argument('--run',required=True,type=int)
    merger.add_argument('--commit',required=True)
    merger.add_argument('--out',required=True,type=Path)
    merger.add_argument('archives',nargs='*',type=Path)
    for item in (verify,merger):
        item.add_argument('--plan',type=Path,default=HERE/'protocol.json')
    args=parser.parse_args()
    if args.mode=='merge':
        sys.exit(0 if merge(args) else 1)
    result=verify_triple(args.directory,args.seed,args.seat,args.plan)
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    sys.exit(int(bool(result['verification_errors'])))
