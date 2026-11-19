"""Reapply archived native actions to the official rules and verify conditional terminal labels."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import subprocess

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parent/'inprocess_snapshot')]
import play
import official_state
import student_state
from verify import clocks
from fixture import public_hashes

PLAN_SHA='013c21283c1e48477ebb56c3668bbf79e4c019b86fd38dfb96f839018e5227bd'
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_bytes())
need=play.need
DATA=HERE/'outputs'
RUNNER_REV='09a635eefb17bb024292c1fe6e8b693dde0bcdc2'
PLAN_REV='4952efa0e6be03d76c38115db60d56db045f87f5'


def read_tape(folder,stem,card):
    payload=(folder/(stem+'.jsonl.gz')).read_bytes()
    raw=gzip.decompress(payload)
    need(sha(payload)==card['turns_sha256'],'Compressed tape mismatch '+stem)
    if 'raw_turns_sha256' in card:
        need(sha(raw)==card['raw_turns_sha256'],'Uncompressed tape mismatch '+stem)
    return [json.loads(line) for line in raw.splitlines()]


def search_stats(rows):
    counts=Counter()
    depth=Counter()
    cand=Counter()
    multi=positive=0
    for row in rows:
        info=row['original']['search']
        n=info['candidates'];d=info['depth'];c=info['counts']
        need(isinstance(n,int) and n>=1 and isinstance(d,int) and d>=0 and info['rolls']>=0,'Invalid original workload')
        need(len(info['measured_cost'])==4 and all(math.isfinite(v) and v>=0 for v in info['measured_cost']),'Invalid cost telemetry')
        if n==1:
            need(d==0 and c['single']==1 and c['searched']==0 and not c['aborted'],'Invalid singleton observer')
        else:
            multi+=1;positive+=int(d>0)
            need(c['single']==0 and ((d>0 and c['searched']==1) or
                 (d==0 and c['searched']==0 and c['aborted']==1)),'Invalid multi-candidate observer')
        counts.update(c);depth[d]+=1;cand[n]+=1
    return dict(total=len(rows),multi_candidate=multi,positive_depth_multi=positive,
                positive_depth_fraction=positive/multi if multi else None,
                meets_minimum_80_percent=(5*positive>=4*multi) if multi else None,
                original_counters=dict(counts),depth_histogram=dict(depth),candidate_histogram=dict(cand))


def combine_clocks(prefix,suffix):
    result={}
    for name in ('student','original'):
        calls=[r[name] for r in prefix+suffix]
        need(all(math.isfinite(c[k]) and c[k]>=0 for c in calls for k in ('cpu','wall')),'Nonfinite public clock')
        stats=clocks(calls);stats['healthy']=stats['hard']==0 and stats['slow']<=3 and stats['overage']<=5
        result[name]=stats
    return result


def served(actor,obs,cfg,order):
    # MenuDraw reconstructs the two actual complete commands. Preserve the
    # unrelated offline generator handle so the warmed actor can be snapshotted.
    rng=actor.mod._behavior_rng
    try:
        return play.audit_menu.per_order(actor,'candidate',obs,cfg,order)
    finally:
        if obs['step']!=0:actor.mod._behavior_rng=rng


def replay_segment(rows,env,live,cfg,actor,bank,side,terrain,forced=None):
    actions=[]
    for index,row in enumerate(rows):
        t=row['step']
        need(not env.done and env.states[0].observation.step==t and public_hashes(live,env,bank)==row['public_before_by_seat'],'Wrong initial public observation')
        cmds=row['orders'];own=row['student'];orig=row['original']
        need(len(cmds)==2 and cmds[side]==own['order'] and cmds[1-side]==orig['order'],'Physical actor role')
        pub=live.observe(env,side,bank[side]);need('full_carbon' not in pub,'Hidden state in student input')
        event=served(actor,pub,cfg,cmds[side]);stored=row.get('student_event')
        if event and event['supported']:
            need(stored is not None,'Missing genuine support')
            for src,dst in [('options','options'),('p','probabilities'),('pick','choice'),('logp','logp'),
                            ('native_logits','native_logits'),('legal','legal'),('prefix','prefix'),('route','route'),
                            ('slot','slot'),('actions','actions')]:
                need(stored[src]==event[dst],'Student '+src+' does not reproduce at '+str(t))
            need(stored['uid']==str(actor.mod._FEAT.own_uid[event['slot']]) and
                 stored['forced']==(index==0 and forced is not None),'Incorrect force or unit identity')
            if stored['forced']:need(stored['pick']==forced,'Forced wrong alternative')
            else:need(stored['pick']==stored['unforced_choice'],'Post-root behavior not natural')
            actions.append(dict(step=t,stored=stored,issued=cmds[side]))
        else:
            need(stored is None and not(index==0 and forced is not None),'Spurious forced or stochastic event')
        for s,command in ((side,own),(1-side,orig)):
            need(math.isfinite(command['wall']) and command['wall']>=0,'Invalid runtime')
            bank[s]-=max(0,command['wall']-1)
        need(bank==row['bank_after'],'Clock accumulation')
        env.step(cmds)
        need(play.board_hash(env,terrain)==row['board_after_sha256'] and public_hashes(live,env,bank)==row['public_after_by_seat'],'Official transition, IDs or observers diverged')
    return actions


def settle(fixture,plan):
    seed,seat=fixture['seed'],fixture['seat'];terrain=fixture['board_sha256']
    out=DATA/f'{seed}_{seat}'
    prefix=read(out/'prefix.json');claim=read(out/'claim.json');finish=read(out/'finish.json')
    root=dict(seed=seed,seat=seat,terrain=terrain,arm_execution_order=fixture['arm_order'],
              receipt_hashes={k:sha((out/(k+'.json')).read_bytes()) for k in ('prefix','claim','finish')})
    need(prefix.get('seed',seed)==seed and prefix.get('seat',seat)==seat and claim['seed']==seed and claim['seat']==seat,'Wrong fixture')
    need(claim['release']==PLAN_REV and claim['protocol_sha256']==PLAN_SHA and claim['terrain']==terrain and claim['arm_order']==fixture['arm_order']
         and set(claim['source_sha256'])==set(prefix['sources']) and claim['source_sha256']==prefix['sources'],
         'Wrong source signatures or prescribed arm order')
    for name,digest in prefix['sources'].items():
        frozen=subprocess.check_output(['git','-C',str(play.ROOT),'show',RUNNER_REV+':'+name])
        need(sha((play.ROOT/name).read_bytes())==sha(frozen)==digest,'Frozen source drift '+name)
    live_plan=read(HERE.parent/'protocol.json')
    for role in ('original','student'):
        need(sha((play.ROOT/live_plan['actors'][role]['file']).read_bytes())==plan[role+'_sha256'],'Policy identity '+role)
    need(sha((HERE.parent/'inprocess_snapshot/receipt.json').read_bytes())==plan['mechanism_sha256'],'Mechanism identity')
    limits=prefix['environment'];need(limits['local_diagnostic'] and limits['affinity']==[0,1] and
        all(a==[0,1] for a in limits['all_thread_affinities'].values()) and limits['torch_threads']==2 and
        limits['omp']==limits['blas']=='1','Wrong declared resource treatment')
    need(set(prefix['cold'])=={'original','student'} and all(math.isfinite(v) and v>=0 for v in prefix['cold'].values()),'Invalid cold time')
    root.update(cold_load_seconds=prefix['cold'],cold_load_le_20=all(v<=20 for v in prefix['cold'].values()),resources=limits,policy_sources={k:plan[k+'_sha256'] for k in ('original','student')})
    pre=read_tape(out,'prefix',prefix)
    prior=search_stats(pre);root['prefix_search']=prior
    env,cfg,live=play.official(seed,terrain,seat);need(sha(bytes(env.states[0].observation.full_carbon))==terrain,'Wrong physical opening board')
    actor=play.audit_menu.collect.Player(play.student_history.PACK);bank=[5.,5.]
    observed=replay_segment(pre,env,live,cfg,actor,bank,seat,terrain)
    root['prefix_tape_sha256']=prefix['turns_sha256'];root['prefix_turns']=len(pre)
    if prefix['status']!='root':
        need(prefix['status']=='no_registered_supported_root' and not any(a['step'] in plan['probes'] for a in observed),
             'Recorded no-root verdict contradicted by physical student')
        need(env.done and prefix['terminal'] and [q.reward for q in env.states]==prefix['official_scores_by_seat'] and
             finish['status']=='no_registered_supported_root','No-root terminal does not reproduce')
        root.update(eligible=False,status='no_registered_supported_root',finish=finish,official_scores_by_seat=prefix['official_scores_by_seat'])
        return root
    t=env.states[0].observation.step
    need(t==prefix['root_step'] and t in plan['probes'] and not any(a['step'] in plan['probes'] and a['step']<t for a in observed),
         'First available registered decision was skipped')
    need(prefix['probes']==[{'step':x,'supported':x==t} for x in plan['probes'] if x<=t] and
         finish['status']=='paired_raw' and finish['arm_order']==fixture['arm_order'] and finish['root_step']==t and
         finish['final_root_restored'],'Probe order/final restoration evidence')
    fingerprints=prefix['root_fingerprints']
    need(set(fingerprints)=={'official','original','student'} and all(len(v)==64 for v in fingerprints.values()),'Missing private-state receipts')
    root_board=play.board_hash(env,terrain);pub=public_hashes(live,env,bank)
    need(root_board==prefix['root_board_sha256'] and pub==prefix['root_public_by_seat'] and
         play.econ(env)==prefix['root_economy'],'Official pre-action boundary')
    saved_env=official_state.take(env,bank);saved_actor=student_state.snapshot(actor)
    branches={};root_events={}
    for idx in fixture['arm_order']:
        name=('A','B')[idx]
        env_bank=official_state.restore(env,saved_env);student_state.restore(actor,saved_actor)
        card=read(out/(name+'.json'));rows=read_tape(out,name,card)
        need(card['root_fingerprints']==fingerprints and card['root_board_sha256']==root_board and
             card['root_public_by_seat']==pub and rows[0]['step']==t,'Reported branch does not restart at shared boundary')
        events=replay_segment(rows,env,live,cfg,actor,env_bank,seat,terrain,idx)
        root_event=events[0]['stored'];root_events[name]=root_event
        need(root_event['options']==prefix['root_orders'] and card['replayed_transitions']==len(rows) and
             card['root_intervention']=={'event':root_event,'orders':rows[0]['orders']} and card['arm']==name and
             card['index']==idx and card['pid']==claim['pid'] and card['sources']==prefix['sources'],
             'Reported source, chosen intervention or CPU process differs')
        for k in ('options','p','unforced_choice','slot','actions','prefix','legal','native_logits','route','uid'):
            need(root_event[k]==prefix['root_menu'][k],'Physical original menu mismatch '+k)
        need(env.done and [s.status for s in env.states]==['DONE','DONE'] and env.states[0].observation.step==card['terminal_step'] and
             [s.reward for s in env.states]==card['official_scores_by_seat'] and play.econ(env)==card['terminal_economy'],'Official terminal outcome')
        scores=[s.reward for s in env.states];m=scores[seat]-scores[1-seat];wdl=float(m>0)+.5*float(m==0)
        u=.5*wdl+.25*(1+math.tanh(m/2000))
        need(abs(card['margin']-m)<=1e-9 and card['wdl']==wdl and abs(card['u']-u)<=1e-12,'Terminal payoff discrepancy')
        branches[name]=dict(tape_sha256=card['turns_sha256'],receipt_sha256=sha((out/(name+'.json')).read_bytes()),
                            steps=len(rows),official_scores_by_seat=scores,terminal_step=card['terminal_step'],
                            root_student_order=rows[0]['orders'][seat],root_original_order=rows[0]['orders'][1-seat],
                            root_choice=idx,score_margin=m,wdl=wdl,u=u,search=search_stats(rows),
                            recorded_prefix_plus_suffix_wall=prefix['prefix_wall_seconds']+card['elapsed_seconds'],
                            recorded_pair_wall_le_1500=prefix['prefix_wall_seconds']+card['elapsed_seconds']<=1500,
                            full_game_clocks=combine_clocks(pre,rows),student_binary_roots=[v['step'] for v in events])
    a,b=branches['A'],branches['B']
    keys=('options','p','unforced_choice','slot','actions','prefix','legal','native_logits','route','uid')
    need(all(root_events['A'][k]==root_events['B'][k] for k in keys),'Actions did not come from a common policy root')
    eligible=dict(distinct_student_orders=a['root_student_order']!=b['root_student_order'],
                  matching_root_state=True,physical_root_examiner_order_equal=a['root_original_order']==b['root_original_order'],
                  official_terminals_replayed=True,native_clocks_healthy=all(v['healthy'] for arm in branches.values() for v in arm['full_game_clocks'].values()),
                  local_cold_and_full_wall_within_limits=root['cold_load_le_20'] and all(v['recorded_pair_wall_le_1500'] for v in branches.values()),
                  multi_search_prefix_80=prior['meets_minimum_80_percent'],multi_search_A_80=a['search']['meets_minimum_80_percent'],
                  multi_search_B_80=b['search']['meets_minimum_80_percent'])
    need(finish['root_examiner_commands_equal']==eligible['physical_root_examiner_order_equal'] and
         abs(finish['second_minus_first_margin']-b['score_margin']+a['score_margin'])<1e-9 and
         abs(finish['second_minus_first_u']-b['u']+a['u'])<1e-12,'Finish tally discrepancy')
    root.update(status='paired_at_terminal',root_step=t,root_board_sha256=root_board,root_public=pub,
                native_origin_fingerprints=fingerprints,root_actual_menu=root_events['A']['options'],root_probabilities=root_events['A']['p'],
                branches=branches,eligibility_conditions=eligible,eligible=all(value is True for value in eligible.values()),
                score_margin_difference_B_minus_A=b['score_margin']-a['score_margin'],delta_u_B_minus_A=b['u']-a['u'])
    return root


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--check-seed',type=int);args=cli.parse_args()
    path=HERE/'protocol.json';need(sha(path.read_bytes())==PLAN_SHA,'Protocol changed')
    plan=read(path);rows=[]
    for f in plan['fixtures']:
        if args.check_seed is not None and f['seed']!=args.check_seed:continue
        try:rows.append(settle(f,plan))
        except Exception as e:
            if args.check_seed is not None:raise
            rows.append(dict(seed=f['seed'],seat=f['seat'],eligible=False,status='unverified',
                             error=type(e).__name__+': '+str(e)))
    if args.check_seed is not None:print(json.dumps(rows));return
    valid=[r for r in rows if r['eligible']]
    res=dict(schema='sapling-real-root-induced-terminal-pairs/1',protocol_sha256=PLAN_SHA,
             verifier_sha256=sha(Path(__file__).read_bytes()),eligible_pairs=len(valid),registered_roots=len(rows),
             eligible_B_positive_delta_u=sum(r['delta_u_B_minus_A']>0 for r in valid),
             eligible_B_negative_delta_u=sum(r['delta_u_B_minus_A']<0 for r in valid),
             eligible_ties_delta_u=sum(r['delta_u_B_minus_A']==0 for r in valid),roots=rows,
             counted_whole_game_wins=0,formal_games=0,
             scope='Previously exposed training maps; one natural prefix and one conditional terminal draw per available arm. Runtime snapshots certify reported original private history at restoration; publicly observable board, actions, score, and student support are independently reconstructed. Host clock and examiner response can still differ between subsequent plies.')
    if any(r['status']=='unverified' for r in rows):
        raise ValueError('Unverified evidence: '+str([(r['seed'],r.get('error')) for r in rows if r['status']=='unverified']))
    play.save(HERE/'settlement.json',res);print(json.dumps([(r['seed'],r['status'],r['eligible']) for r in rows]))


if __name__=='__main__':main()
