"""Replay the registered continuous curriculum against archived original actions, with no new games."""
import os
for var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[var]='1'
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import play as runner
import student_history

HERE=Path(__file__).resolve().parent
ROOT=runner.ROOT
need=runner.need
sha=lambda b: hashlib.sha256(b).hexdigest()
read=lambda p: json.loads(p.read_bytes())


def clocks(items):
    walls=[d['wall'] for d in items]
    return dict(calls=len(items),wall=sum(walls),cpu=sum(d['cpu'] for d in items),
                p95=float(np.percentile(walls,95)),p99=float(np.percentile(walls,99)),
                maximum=max(walls),slow=sum(t>1 for t in walls),hard=sum(t>3 for t in walls),
                overage=sum(max(0,t-1) for t in walls))


def check_source(seed,seat):
    folder=HERE/'outputs'/f'{seed}_{seat}'
    claim=read(folder/'claim.json');prefix=read(folder/'prefix.json');suffix=read(folder/'unforked.json')
    plan=read(HERE/'protocol.json')
    sources={r:sha((ROOT/x['file']).read_bytes()) for r,x in plan['actors'].items()}
    authority=subprocess.check_output(['git','-C',str(ROOT),'show',runner.RULECOMMIT+':sources/master3s2/master3s2.py'])
    need(sources['original']==sha(authority) and sources=={r:x['sha256'] for r,x in plan['actors'].items()},'Native original or policy source')
    need(all(q['seed']==seed and q['seat']==seat and q['source_sha256']==sources for q in (claim,prefix,suffix)),'Wrong role / declared source')
    need(sha((HERE/'protocol.json').read_bytes())==claim['protocol_sha256']==runner.PROTO and
         claim['runner_sha256']==sha((HERE/'play.py').read_bytes()),'Unsigned runtime')
    need(read(folder/'branches.json')=={'unforked':0} and prefix['status']=='root' and
         suffix['arm']=='unforked' and prefix['source_pid']==suffix['source_pid']==suffix['pid']==claim['pid'] and
         suffix['boundary_before']==prefix['boundary'],'Cold or duplicated continuation')
    terrain=read(HERE/'clearance.json')['by_seed'][str(seed)]
    need(terrain==claim['terrain']==prefix['terrain'] and not claim['preflight'],'Wrong map')
    env,cfg,live=runner.official(seed,terrain,seat)
    need(sha(bytes(env.states[0].observation.full_carbon))==terrain,'Opening map mismatch')
    actor=runner.audit_menu.collect.Player(ROOT/plan['actors']['student']['file'])
    bank=[5.,5.];depth=Counter();sizes=Counter();search_counts=Counter()
    times={'prefix_original':[],'suffix_original':[],'prefix_teacher':[],'prefix_shadow':[],'suffix_student':[]}
    mismatch=0;natural_events=[];files={};future_changes=0
    for part,card,role in [('prefix',prefix,'apex'),('unforked',suffix,'student')]:
        raw=(folder/(part+'.jsonl.gz')).read_bytes();expanded=gzip.decompress(raw)
        need(sha(raw)==card['turns_sha256'] and sha(expanded)==card['raw_turns_sha256'],'Action tape digest changed')
        files[part]=sha(raw)
        active_rows=[];opp_rows=[];diff=0;scheduled=0;supported=0
        for line in expanded.splitlines():
            row=json.loads(line);t=row['step']
            need(not env.done and t==env.states[0].observation.step and
                 row['active_role']==role and ((role=='apex')==(t<240)),'Discontinuous physical source')
            pub=[live.observe(env,p,bank[p]) for p in (0,1)]
            need(all('full_carbon' not in q for q in pub) and
                 [sha(live.encoded(q)) for q in pub]==row['public_sha256_by_seat'],'Actor-visible state mismatch')
            calls=row['calls'];orders=row['orders'];active=calls['active'];opp=calls['original']
            need(len(orders)==2 and orders[seat]==active['order'] and orders[1-seat]==opp['order'],'Issued roles mismatch')
            need(set(calls)==({'original','active','shadow'} if role=='apex' else {'original','active'}),'Unexpected shadow/active role')
            for k,call in calls.items():
                need(all(math.isfinite(call[f]) and call[f]>=0 for f in ('cpu','wall')) and call['core']==(1 if k=='original' else 0),'Invalid call clock/core')
            for s,item in [(seat,active),(1-seat,opp)]:bank[s]-=max(0.,item['wall']-1)
            need(bank==row['overage_after'],'Fast-step overage refund or missing clock')
            sc=opp['search'];need(sc['counts']['single']+sc['counts']['searched']==1,'Original search missing')
            need(sc['depth']>=0 and sc['candidates']>=1 and sc['rolls']>=0 and len(sc['measured_cost'])==4 and
                 all(math.isfinite(v) and v>=0 for v in sc['measured_cost']),'Invalid original search telemetry')
            depth[sc['depth']]+=1;sizes[sc['candidates']]+=1;search_counts.update(sc['counts'])
            history=calls['shadow'] if role=='apex' else active
            act=runner.audit_menu.per_order(actor,'candidate',pub[seat],cfg,history['order'])
            if role=='apex':
                x=active['order']!=history['order'];mismatch+=x;diff+=x
                times['prefix_teacher'].append(active);times['prefix_shadow'].append(history);times['prefix_original'].append(opp)
            else:
                times['suffix_student'].append(active);times['suffix_original'].append(opp)
                if t in runner.audit_menu.STEPS:
                    scheduled+=1;observed=active['event']
                    need(act is not None and bool(observed is not None)==act['supported'],'Recorded support mismatches served menu')
                    if observed:
                        for orig,new in [('p','probabilities'),('pick','choice'),('options','options'),('logp','logp'),
                                         ('slot','slot'),('route','route'),('prefix','prefix'),('actions','actions'),
                                         ('legal','legal'),('native_logits','native_logits')]:
                            need(observed[orig]==act[new],'Recorded action or conditional probabilities differ')
                        need(observed['uid']==str(actor.mod._FEAT.own_uid[act['slot']]),'Focal worker identity mismatch')
                        supported+=1;future_changes+=act['choice']
                        natural_events.append(dict(step=t,uid=observed['uid'],chosen=act['choice'],probabilities=act['probabilities'],issued=act['issued']))
                else:need('event' not in active,'Unexpected natural event')
            active_rows.append(active);opp_rows.append(opp)
            env.step(orders)
        stats=card['stats'];need(stats['steps']==len(active_rows) and stats['max_wall_active']==max(d['wall'] for d in active_rows) and
            stats['max_wall_original']==max(d['wall'] for d in opp_rows),'Segment clock aggregate')
        for side,cl in [(seat,active_rows),(1-seat,opp_rows)]:
            need(stats['slow_'+str(side)]==sum(x['wall']>1 for x in cl) and stats['hard_'+str(side)]==sum(x['wall']>3 for x in cl),'Slow/hard count differs')
        if role=='apex':
            b=prefix['boundary']
            need(stats['shadow_disagreements']==diff and env.states[0].observation.step==b['step']==prefix['reached']==240 and
                 runner.board_hash(env,terrain)==b['board_sha256'] and runner.econ(env)==b['economy'] and bank==b['overage_by_seat']
                 and [sha(live.encoded(live.observe(env,p,bank[p]))) for p in (0,1)]==b['public_sha256_by_seat']
                 and sc['measured_cost']==b['examiner_cost'] and student_history.snapshot(actor)==b['student_history_sha256'],'Root physical, private receipt or student history mismatch')
            for k in ('tracker_sha256','prev_players_sha256','examiner_rng_sha256','examiner_search_stats_sha256','student_rng_sha256'):
                digest=b[k];need(len(digest)==64 and int(digest,16)>=0,'Invalid captured private state identity')
        else:need(scheduled==stats['scheduled'] and supported==stats['supported'],'Natural choice counting mismatch')
    need(suffix['first_step']==240 and suffix['terminal_step']==env.states[0].observation.step and env.done and suffix['terminal']
         and suffix['terminal_status']==[x.status for x in env.states]==['DONE','DONE']
         and suffix['official_scores_by_seat']==[x.reward for x in env.states] and suffix['terminal_economy']==runner.econ(env)
         and suffix['overage_by_seat']==bank,'Official termination/score mismatch')
    scores=suffix['official_scores_by_seat'];margin=scores[seat]-scores[1-seat]; wdl=float(margin>0)+.5*float(margin==0)
    start_cash=b['economy']['cash'];start_margin=start_cash[seat]-start_cash[1-seat]
    return dict(seed=seed,seat=seat,terrain=terrain,source_pid=claim['pid'],archive_sha256=files,
        prefix_steps=prefix['reached'],suffix_steps=suffix['stats']['steps'],shadow_disagreements=mismatch,
        verified_handoff_feature_sha256=b['student_history_sha256'],board_sha256=b['board_sha256'],
        original_private_state_receipt_continuous=True,official_scores_by_seat=scores,root_cash_by_seat=start_cash,
        root_cash_margin=start_margin,terminal_margin=margin,margin_change_from_root=margin-start_margin,
        wdl=wdl,u=.5*wdl+.25*(1+math.tanh(margin/2000)),natural_events=natural_events,natural_alternatives=future_changes,
        wall_cpu={k:clocks(v) for k,v in times.items()},original_depth_histogram=dict(depth),
        original_candidates_histogram=dict(sizes),original_search_counts=dict(search_counts))


def main():
    command=argparse.ArgumentParser(description=__doc__);command.add_argument('--check-seat',type=int,choices=(0,1));opt=command.parse_args()
    diag=read(HERE/'mechanism/diagnosis.json')
    for k,v in diag['fixture_files_sha256'].items():need(sha((HERE/'mechanism'/k).read_bytes())==v,'Fixture identity')
    need(diag['child_exit']==-9 and read(HERE/'mechanism/branches.json')=={'draw_a':-9} and diag['registered_map_exposures']==0,'Fallback lacks reproducer')
    if opt.check_seat is not None:
        print(json.dumps(check_source(621758480,opt.check_seat),ensure_ascii=False));return
    rows=[check_source(621758480,side) for side in (0,1)]
    doc=dict(schema='sapling-live-apex-rollin-independent-verification/1',verifier_sha256=sha(Path(__file__).read_bytes()),
        runner_sha256=sha((HERE/'play.py').read_bytes()),protocol_sha256=runner.PROTO,unforked=True,games=rows,
        verified_official_transitions=sum(r['prefix_steps']+r['suffix_steps'] for r in rows),new_games=0,
        scope='Recorded physical actions reconstruct both official complete environments and scores. Student shadow and suffix choices replay from the genuine parent package. Replayed private student history, compared original RNG/tracker/cost/search snapshots from the unchanged live process at the boundary; independent replay does not recreate hidden RNG values or original invocation wall times. Timings are instrumented local training diagnostics. No independent counterfactual, whole-game Sapling opponent result or ladder resource certificate.')
    runner.save(HERE/'verification.json',doc)
    print(json.dumps([(r['seat'],r['terminal_margin']) for r in rows]))


if __name__=='__main__':main()
