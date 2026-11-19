"""Idempotent owner of the separately preregistered training supplement; no development or formal games."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent
sys.path.insert(0,str(TRAIN))
import batch_cli as engine
from continue_intakes import active_lock,verify_sealed
from select_roots import load,retain,sha


def registered():
    d=load(HERE/'registration.json')
    anchors={HERE/'register.py':'register_program_sha256', TRAIN/'protocol.json':'original_protocol_sha256',
             TRAIN/'next_intake.json':'original_queue_sha256',TRAIN/'batches/intake_03/aggregate.json':'original_03_aggregate_sha256'}
    if not all(sha(p)==d[k] for p,k in anchors.items()) or d['original_05_must_seal'] is not True:
        raise ValueError('supplement registration or original frozen anchors have changed')
    return d


def prior_dirs(reg,upto):
    dirs=[TRAIN/z for z in reg['expected_original_05_prefix']]
    for item in reg['batches']:
        if item['name']==upto:
            return dirs
        dirs.append(HERE/'batches'/item['name'])
    if upto is None:
        return dirs
    raise ValueError('unregistered batch')


def signature(areas):
    all_rows=[];boards=set();chain=[];windows=0;empty=[]
    for area in areas:
        rows,source=verify_sealed(area)
        if len(source)!=len(set(source.values())) or boards&set(source.values()):
            raise ValueError('training opening has already been exposed')
        boards.update(source.values());all_rows.extend(rows)
        chain.append({'dir':str(area.relative_to(TRAIN)),'labels_sha256':sha(area/'training_labels.json')})
        chosen=load(area/'selection.json');windows+=len(chosen['windows']);empty.extend(chosen['empty_windows'])
        if area==TRAIN:
            continue
        s=load(area/'aggregate.json')
        if (s['source_intakes']!=chain or s['distinct_maps']!=len(boards) or s['complete_pairs']!=len(all_rows)
                or s['filled_pairs']!=sum(r['pay_intent_filled'] for r in all_rows) or s['held_out_games']!=0):
            raise ValueError('incorrect, incomplete or substituted aggregate')
    paid=[r for r in all_rows if r['pay_intent_filled']]
    positive={r['seed'] for r in paid if r['target_delta_official_margin']>0}
    negative={r['seed'] for r in paid if r['target_delta_official_margin']<0}
    return {'distinct_source_maps':len(boards),'sealed_pairs':len(all_rows),'actual_paid_pairs':len(paid),
            'positive_maps':sorted(positive),'negative_maps':sorted(negative),'planned_windows':windows,
            'empty_windows':empty,'source_intakes':chain}


def shortfall(counts,goals):
    return max(0,goals['independent_training_maps']-counts['distinct_source_maps']),max(0,goals['completed_actually_paid_pairs']-counts['actual_paid_pairs'])


def status():
    reg=registered();first=reg['batches'][0]['name'];base=prior_dirs(reg,first)
    if not (base[-1]/'aggregate.json').is_file():
        if any((HERE/'batches'/r['name']/'protocol.json').exists() for r in reg['batches']):
            raise ValueError('supplement has registration before original fifth intake seal')
        return {'phase':'waiting_for_original_05_seal','next':None,'training_started':False,
                'original_05_seal':str((base[-1]/'aggregate.json').relative_to(TRAIN))}
    areas=list(base);counts=signature(areas);pending=None;used=[]
    for entry in reg['batches']:
        wd=HERE/'batches'/entry['name']
        if shortfall(counts,reg['goals'])==(0,0):
            if (wd/'protocol.json').exists():raise ValueError('sampling continued beyond predeclared support')
            continue
        if not (wd/'aggregate.json').is_file():
            pending=entry['name'];break
        areas.append(wd);counts=signature(areas);used.append(entry['name'])
    maps,paid=shortfall(counts,reg['goals'])
    signed=reg['goals']['minimum_distinct_maps_per_sign']
    enough_signs=len(counts['positive_maps'])>=signed and len(counts['negative_maps'])>=signed
    if pending:
        phase='collect_next_complete_training_block'
    elif maps or paid:
        phase='all_registered_blocks_exhausted'
    else:
        phase='registered_count_target_reached'
    return {'phase':phase,'next':pending,'completed_supplement_blocks':used,'counts':counts,
            'missing_training_maps':maps,'missing_paid_pairs':paid,'adequate_effect_sign_coverage':enough_signs,
            'support_for_fit':not (maps or paid) and enough_signs,'held_out_games':0}


def issue(name,reg):
    entry=next(r for r in reg['batches'] if r['name']==name)
    prior=prior_dirs(reg,name)
    state=status()
    if state['phase']=='waiting_for_original_05_seal' or (name!=state['next'] and name not in state.get('completed_supplement_blocks',())):
        raise ValueError('previous intakes incomplete, count reached or wrong block')
    wd=HERE/'batches'/name;wd.mkdir(exist_ok=True,parents=True)
    with (wd/'.registration.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        plan_path=wd/'protocol.json'
        frozen=load(TRAIN/'batches/intake_03/protocol.json')
        build={**frozen,'schema':'sapling-tree-response-training-supplement-batch-v1','intake_name':name,
               'queue_sha256':sha(HERE/'registration.json'),'train_maps':entry['maps'],'training_job_registry':entry['jobs'],
               'prior_receipts':{str(q.relative_to(TRAIN)):sha(q/'training_labels.json') for q in prior},
               'selector_at_registration_sha256':reg['selector_at_registration_sha256'],
               'shared_decode_at_registration_sha256':reg['shared_decode_at_registration_sha256']}
        if plan_path.is_file():
            plan=load(plan_path)
            if any(plan.get(k)!=v for k,v in build.items() if k not in ('source_revision','registered_utc')):
                raise ValueError('different intake policy, maps, methods, jobs or earlier label seals')
        else:
            build['source_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=engine.ROOT,text=True).strip()
            build['registered_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            retain(plan_path,build);plan=build
        resources=load(TRAIN/'response_resources.json')
        retain(wd/'response_resources.json',{**resources,'parent_protocol_sha256':sha(plan_path),
                                             'data_parent':sha(TRAIN/'response_resources.json'),'source_maps':entry['maps']})
    return wd,plan,prior


def drain():
    reg=registered()
    with (HERE/'.owner.lock').open('a') as fd:
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('supplement already supervised',flush=True);return
        signed=prior_dirs(reg,reg['batches'][0]['name'])[-1]/'aggregate.json'
        if not signed.exists():
            print('waiting for verified original intake 05',flush=True)
        while not signed.exists():time.sleep(20)
        # Shared across original and additional intake ownership. No second native court opens.
        with (TRAIN/'.continuation.lock').open('a') as global_owner:
            fcntl.flock(global_owner,fcntl.LOCK_EX)
            while True:
                state=status();print(json.dumps(state,ensure_ascii=False),flush=True)
                if state['next'] is None:
                    retain(HERE/'completion.json',{'registration_sha256':sha(HERE/'registration.json'),**state})
                    return
                name=state['next'];wd,plan,previous=issue(name,reg);engine.modules(wd,plan)
                from priority_slot import wait_before_source
                engine.run_intake(wd,plan,clearance_script=HERE/'clearance.py',entrypoint=Path(__file__),
                                  previous=previous,before_source=wait_before_source)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command')
    parser.add_argument('mode',nargs='?',choices=('arm',));parser.add_argument('--index',type=int);parser.add_argument('--arm',choices=('parent','alternative'))
    a=parser.parse_args()
    if a.command=='status':
        print(json.dumps(status(),ensure_ascii=False,indent=2));return
    if a.command=='drain':
        drain();return
    if a.mode!='arm' or a.index is None or a.index<0 or a.arm is None:
        parser.error('use status, drain, or registered-intake arm with index and native arm')
    wd,plan,_=issue(a.command,registered());engine.modules(wd,plan)
    case=wd/engine.responder.records()[a.index]['case_directory']
    with (case/f'.response_{a.arm}.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);engine.responder.play(a.index,a.arm)


if __name__=='__main__':
    main()
