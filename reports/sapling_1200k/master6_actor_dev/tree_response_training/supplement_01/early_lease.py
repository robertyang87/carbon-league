"""Conditional four-root native lease between supplementary training blocks six and seven."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sys
import time

from priority_slot import TRAIN,ROOT,HERE,sha,load,retain,check
from continue_intakes import verify_sealed

DEV=TRAIN.parent/'early_lane_dev'
TRIAL=DEV/'native_response'
SLOT=HERE/'early_lane_slot'
RULE='1a461718ed0e0e161838cc5bde1592ed34cb66bde94893d135b3d5bd4b023db8'


def registered(parity_pass=True):
    plan=load(TRIAL/'protocol.json');r=load(SLOT/'reservation.json')
    check(sha(TRIAL/'protocol.json')==RULE==r['native_protocol_sha256'] and plan['protocol_sha256']==sha(DEV/'protocol.json'),
          'wrong frozen tactic, scoring rule or branch list')
    p=load(DEV/'parity.json')
    check(p['candidate_sha256']==plan['candidate_sha256']==sha(DEV/'candidate.py')
          and p['parent_sha256']==plan['compared_parent_sha256']==load(TRAIN/'protocol.json')['parent_pack_sha256']
          and p['weights_sha256']==plan['weights_sha256']==sha(DEV/'weights.json')
          and p['gate_sha256']==plan['gate_sha256']==sha(DEV/'gate.json'),'different candidate, parent or rule')
    by_map={}
    for b in ('intake_04','intake_05'):
        for g in load(TRAIN/'batches'/b/'source_receipt.json')['games']:
            by_map[g['seed'],g['seat']]=g
    if parity_pass:
        need={'all_order_agreements':2392,'v1_order_agreements':2392,'candidate_rows_and_scores_agree':464,'overrides_agree':41}
        actual=dict.fromkeys(need,0)
        check(p['result']=='pass' and set(p['evidence'])=={f'{s}:{v}' for s,v in by_map},'missing preregistered public equivalence')
        for (s,v),g in by_map.items():
            path=DEV/'parity_rows'/f'{s}_{v}.json';trace=load(path)
            check(sha(path)==p['evidence'][f'{s}:{v}'] and trace['source_sha256']==g['raw_sha256']
                  and trace['map_sha256']==g['map_sha256'] and trace['candidate_sha256']==p['candidate_sha256'],'equivalence evidence changed')
            for k in actual:actual[k]+=trace['counts'][k]
        check(actual==need==p['total'],'public runtime failed parity')
    check(len(plan['roots'])==len({(q['seed'],q['seat'],q['t']) for q in plan['roots']})==4,'need four preregistered roots')
    for q in plan['roots']:
        g=by_map[q['seed'],q['seat']]
        check(g['raw_sha256']==q['source_sha256']==sha(ROOT/q['source']) and g['raw_path']==q['source'],'source game differs')
    return r,plan,p


def make_grant():
    sixth=HERE/'batches/intake_06'
    if not (sixth/'aggregate.json').exists() or not (DEV/'parity.json').exists():return None
    if (HERE/'batches/intake_07/source/results.json').exists():raise ValueError('priority requested after next native intake opened')
    with (SLOT/'.state.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        file=SLOT/'granted.json'
        if not file.exists():verify_sealed(sixth)
        reg,plan,_=registered()
        rec={'reservation_sha256':sha(SLOT/'reservation.json'),'protocol_sha256':RULE,'intake_06_aggregate_sha256':sha(sixth/'aggregate.json'),
             'parity_sha256':sha(DEV/'parity.json'),'candidate_sha256':plan['candidate_sha256'],'cpus':reg['native_cpus']}
        retain(file,rec)
    return file


@contextmanager
def native_session():
    while not make_grant():time.sleep(8)
    with (SLOT/'.native-owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(not any((SLOT/f).exists() for f in ('released.json','skipped.json')),'early lane slot already closed')
        yield


def terminal(official):
    _,plan,_=registered();summary=load(TRIAL/'result.json')
    check(summary['status']=='complete' and summary['completed_pairs']==4 and summary['completed_arms']==8
          and summary['protocol_sha256']==RULE and summary['candidate_sha256']==plan['candidate_sha256']
          and summary['parity_sha256']==sha(DEV/'parity.json'),'unexpected branch completion')
    linked=summary['files_sha256']
    for field,h in linked.items():
        path=(TRIAL/field).resolve()
        check(path.is_relative_to(TRIAL.resolve()) and sha(path)==h,'invalid signed response file')
    cases=[]
    for q in plan['roots']:
        s=q['seat'];d=TRIAL/f"case_{q['seed']}_{s}";proof=load(d/'verified.json.gz')
        raws=load(ROOT/q['source']);start=q['t'];shared=None
        for name in ('parent','alternative'):
            native=d/f'live_{name}.json';g=load(native);actions=d/g['action_tape'];tail=load(actions)
            v=proof['arms'][name];states=g['score']
            check(all(x.relative_to(TRIAL).as_posix() in linked for x in (native,actions,d/'verified.json.gz'))
                  and sha(actions)==g['action_sha256']==v['action_sha256'] and g['statuses']==v['statuses']==['DONE','DONE']
                  and g['source_sha256']==plan['examiner_sha256'] and g['original_trajectory_sha256']==q['source_sha256']
                  and g['seed']==q['seed'] and g['seat']==s and g['search_calls']==len(tail)-1
                  and states==v['score_by_physical_seat'] and g['root']['t']==start,'mismatched native arm or official score')
            stream=g['shared_init_sha256']
            check((shared is None or stream==shared) and stream==sha(d/'live_shared_init.json'),'mismatched opponent response stream')
            shared=stream
            check(tail[0][1-s]==raws['actions'][start][1-s],'opponent saw the intervention on the common root')
            if official:
                from select_roots import live
                env=live.begin(raws)
                for pair in raws['actions'][:start]+tail:
                    check(not env.done,'extra move after true terminal')
                    env.step(pair)
                check(env.done and [r.status for r in env.states]==g['statuses'] and [r.reward for r in env.states]==states,'independent official replay mismatch')
        cases.append({'seed':q['seed'],'seat':s,'root':start,'proof_sha256':sha(d/'verified.json.gz')})
    return {'grant_sha256':sha(SLOT/'granted.json'),'result_sha256':sha(TRIAL/'result.json'),'original_responses':8,'cases':cases,'decision':summary['decision']}


def release():
    with (SLOT/'.state.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=SLOT/'released.json';check(not (SLOT/'skipped.json').exists(),'early lane was skipped')
        retain(receipt,terminal(official=not receipt.exists()))
        return receipt


def skip():
    _,_,p=registered(parity_pass=False);f=DEV/'early_lane_skip.json';skip=load(f)
    check(p['result']!='pass' and skip['native_games']==0 and skip['protocol_sha256']==RULE and skip['parity_sha256']==sha(DEV/'parity.json')
          and skip['reason'] and not any(TRIAL.glob('case_*/live_*_actions.json.gz')) and not any(TRIAL.glob('case_*/live_*_checkpoint.json.gz')),
          'cannot skip a passed equivalence or played response')
    retain(SLOT/'skipped.json',{'protocol_sha256':RULE,'parity_sha256':sha(DEV/'parity.json'),'skip_sha256':sha(f),'native_arms':0})
    return SLOT/'skipped.json'


def wait_before_source(wd,plan):
    if wd.resolve()!=HERE/'batches/intake_07':return
    while True:
        if (SLOT/'released.json').exists():
            check(load(SLOT/'released.json')==terminal(False),'different early lane release');return
        if (SLOT/'skipped.json').exists():skip();return
        make_grant()
        time.sleep(9)
