"""Lease the existing native CPU pair after registered map/last-action and tree-head studies."""
from native import (case_dir, check, datetime, ENV, fcntl, freeze, HERE, HEAD, live, native_module,
                    os, OUT, Path, read, ROOT, sha, subprocess, sys, T, ThreadPoolExecutor, time)

SUP=T/'supplement_01'
EIGHT=SUP/'batches/intake_08'
LAST=HERE.parent/'endgame_queue_dev'


def first_window():
    while not (SUP/'completion.json').is_file():time.sleep(5)
    final=read(SUP/'completion.json')
    check((EIGHT/'aggregate.json').is_file() and final['phase'] in ('registered_count_target_reached','all_registered_blocks_exhausted')
          and final['counts']['source_intakes'][-1]['labels_sha256']==sha(EIGHT/'training_labels.json'),'eighth half-finished or different data cutoff')
    with (T/'.continuation.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        grant=SUP/'endgame_queue_grant.json'
        freeze(grant,{'cpu':0,'completion_sha256':sha(SUP/'completion.json'),'permitted_native_calls':8,
                      'last_move_protocol_sha256':sha(LAST/'native_protocol.json'),'eighth_source_sha256':sha(EIGHT/'aggregate.json')})
        while not (LAST/'native_result.json').is_file():time.sleep(3)
        outcome=read(LAST/'native_result.json')
        check(outcome['native_calls']==8 and len(outcome['pairs'])==4 and outcome['grant_sha256']==sha(grant)
              and outcome['protocol_sha256']==sha(LAST/'native_protocol.json'),'last-shot native receipt mismatch')
    while not (HEAD/'independent_acceptance.json').is_file():time.sleep(7)
    accepted=read(HEAD/'independent_acceptance.json')
    check(accepted['completion_sha256']==sha(SUP/'completion.json') and accepted['passed']==read(HEAD/'status.json')['eligible_for_development'],
          'tree-head acceptance detached from training')
    result={'tree_08_completion_sha256':sha(SUP/'completion.json'),'shared_final_probe_sha256':sha(LAST/'native_result.json'),
            'tree_head_acceptance_sha256':sha(HEAD/'independent_acceptance.json'),'tree_head_precedes_hire':accepted['passed']}
    if accepted['passed']:
        comp=HEAD/'development_runs/comparison.json'
        while not comp.is_file():time.sleep(7)
        from economy_lease import verify_tree_development
        rpt=read(comp);check(rpt['games']==8 and not rpt['formal'],'incomplete registered tree development')
        verify_tree_development(comp.parent,rpt);result['tree_full_development_sha256']=sha(comp)
    return result


def certify(row):
    wd=case_dir(row);raw=read(ROOT/row['source']);t=row['t'];s=row['seat'];root=row['worker']
    from focal_life import audit_case
    accounting=audit_case(wd)
    official={}
    for a in ('parent','alternative'):
        rec=read(wd/f'live_{a}.json');path=wd/rec['action_tape'];actions=read(path)
        expected=row['root_check'][a]['joint_action']
        check(actions[0]==expected==rec['root_actions'] and actions[0][1-s]==raw['actions'][t][1-s],
              'actual opponent or source-place order differs at intervention')
        check(sha(path)==rec['action_sha256'] and rec['pack_sha256']==read(OUT/'protocol.json')['parent_sha256']
              and rec['source_sha256']==live.MASTER_SHA and rec['shared_init_sha256']==sha(wd/'live_shared_init.json')
              and rec['original_trajectory_sha256']==sha(ROOT/row['source']) and rec['search_calls']==len(actions)-1,
              'native original, response stream or recorded trajectory differs')
        env=live.begin(raw)
        for joint in raw['actions'][:t]+actions:
            check(not env.done,'data contains actions after termination');env.step(joint)
        scores=[x.reward for x in env.states];statuses=[x.status for x in env.states]
        check(env.done and statuses==rec['statuses']==['DONE','DONE'] and scores==rec['score'] and scores[s]-scores[1-s]==rec['margin'],
              'independent official terminal score differs')
        official[a]={'score_by_physical_seat':scores,'margin':rec['margin'],'u':rec['u'],'statuses':statuses,'verified_native_search_calls':rec['search_calls'],
                     'search_completed':rec['search_completed'],'search_changed':rec['search_changed'],'raw_sha256':rec['action_sha256']}
    gains=official['alternative']['margin']-official['parent']['margin']
    collector=accounting['arms']['alternative']['own_focal_deposits']
    freeze(wd/'verified.json.gz',{'official':official,'life':accounting,'seed':row['seed'],'seat':s,'root':t,
                                 'terminal_net_gain':gains,'collector_banked':collector,'registered_payroll':row['price']})
    return {'seed':row['seed'],'seat':s,'turn':t,'net_margin_gain':gains,'collector_deposits':collector,
            'collector_deposits_over_payroll':collector>row['price'],'positive_terminal_delta':gains>0,'proof_sha256':sha(wd/'verified.json.gz')}


def result(plan,rows):
    each_good=all(r['positive_terminal_delta'] and r['collector_deposits_over_payroll'] for r in rows)
    out={'protocol_sha256':sha(OUT/'protocol.json'),'candidate_sha256':plan['candidate_sha256'],'source_parent_sha256':plan['parent_sha256'],
         'native_cases':rows,'all_four_responses':True,'decision':'eligible_for_separate_new_development' if each_good else 'do_not_expand',
         'formal_games':0,'scope':'two selected historical training states against fresh original continuations, one per physical seat'}
    freeze(OUT/'results.json',out)
    print(out,flush=True)


def run():
    sys.path.insert(0,str(SUP));reg=read(OUT/'protocol.json')
    with (OUT/'.schedule.lock').open('a') as job:
        fcntl.flock(job,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initial=first_window()
        with (T/'.continuation.lock').open('a') as resources:
            fcntl.flock(resources,fcntl.LOCK_EX)
            freeze(OUT/'granted.json',{**initial,'cpus':[0,1],'protocol_sha256':sha(OUT/'protocol.json'),
                    'native_runner_sha256':sha(HERE/'native.py'),'scheduler_sha256':sha(Path(__file__))})
            rows=[]
            for i,r in enumerate(reg['cases']):
                wd=case_dir(r);order=('parent','alternative') if i==0 else ('alternative','parent')
                if not (wd/'resource_at_start.json').exists():
                    freeze(wd/'resource_at_start.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roles_by_cpu':dict(enumerate(order)),
                           'quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'load':Path('/proc/loadavg').read_text().strip(),'grant_sha256':sha(OUT/'granted.json')})
                def worker(arm_cpu):
                    cpu,name=arm_cpu
                    subprocess.run(['taskset','-c',str(cpu),sys.executable,'-u',str(HERE/'native.py'),'worker',str(i),name,str(cpu)],env=ENV,check=True)
                with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,enumerate(order)))
                # Complete both preregistered pairs even if external focal bookkeeping is temporarily unavailable.
            for r in reg['cases']:rows.append(certify(r))
            result(reg,rows)
