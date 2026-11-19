"""Run a named registered continuation intake without changing earlier source, training or formal artifacts."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import source_games as sg
import select_roots as selector
import pair_games as responder
import label as evaluator
import opportunities
from batch_summary import report as write_report
from source_games import ROOT,ENV,trial
from select_roots import load,retain,sha

CODE=Path(__file__).resolve().parent

def registry(name):
    queue=load(CODE/'next_intake.json')
    if queue['source_data_sha256']!=sha(CODE/'training_labels.json') or queue['state']!='registered_for_later_not_started':
        raise ValueError('no active continuation or earlier training labels changed')
    if not all(queue['already_verified']['maps_per_realized_effect_sign'].values()):
        raise ValueError('need actual paid positive and negative labels')
    return next(r for r in queue['batches'] if r['name']==name)

def phase_dirs(name):
    batch=registry(name)
    return [CODE]+[CODE/'batches'/r['name'] for r in load(CODE/'next_intake.json')['batches'] if r['name']<batch['name']]

def register(name):
    entry=registry(name)
    wd=CODE/'batches'/entry['name']
    wd.mkdir(parents=True,exist_ok=True)
    with (wd/'.registration.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        return _register(wd,entry)

def _register(wd,entry):
    name=entry['name']
    original=load(CODE/'protocol.json')
    old_dirs=phase_dirs(name)
    if not all((q/'training_labels.json').exists() for q in old_dirs):
        raise ValueError('previous independent intake has not completed')
    for earlier in old_dirs[1:]:
        summary=earlier/'aggregate.json'
        if not summary.is_file():
            raise ValueError('previous intake has not sealed its verified aggregate')
        link=load(summary)['source_intakes'][-1]
        if link['dir']!=str(earlier.relative_to(CODE)) or link['labels_sha256']!=sha(earlier/'training_labels.json'):
            raise ValueError('previous aggregate or labels changed')
    new=set(entry['maps'])
    if new&set(original['development_maps']):
        raise ValueError('reserved development map requested')
    for directory in old_dirs:
        rows=load(directory/'source_receipt.json')['games']
        if any(r['seed'] in new or r['map_sha256'] in entry['terrain_at_queue_time'].values() for r in rows):
            raise ValueError('duplicate historical source map')
    if entry['activation'].startswith('fill_only'):
        labels=[r for q in old_dirs for r in load(q/'training_labels.json')['pairs']]
        if len({r['seed'] for r in labels})>=8 and sum(r['pay_intent_filled'] for r in labels)>=32:
            raise ValueError('fill-only case cannot start once required support is reached')
    expected=wd/'protocol.json'
    if expected.is_file():
        plan=load(expected)
        if plan['queue_sha256']!=sha(CODE/'next_intake.json') or plan['train_maps']!=entry['maps'] or plan['parent_pack_sha256']!=original['parent_pack_sha256']:
            raise ValueError('existing intake was registered under a different source')
    else:
        current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        plan={**original,'source_revision':current,'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'schema':'sapling-tree-response-training-continuation-v1','intake_name':name,
              'queue_sha256':sha(CODE/'next_intake.json'),'prior_receipts':{str(q.relative_to(CODE) or '.'):sha(q/'training_labels.json') for q in old_dirs},
              'train_maps':entry['maps'],'development_maps':original['development_maps'],'training_job_registry':entry['jobs'],
              'strata':'Rank each window by hamming distance to its frozen desired_exposure_bits (rotated within each map), baseline-orientation mismatch, then SHA of tree-pair-v1|seed|seat|step|uid. No terminal or future opponent signals in selection.',
              'responder_rng':'Use exact per-case training_job_registry.rng_seed from the cross-intake registry.',
              'source_math':'A state also carries a map-index-independent risk bit in this intake, so enemy-freight exposure is no longer tied to source-map identity.',
              'resources':'Source games use core 0/1; responses may use two balanced pairs in parallel as registered in response_resources.json. Do not open other new source map blocks concurrently.'}
        plan['selector_at_registration_sha256']=snapshot_selector(Path(__file__))
        plan['shared_decode_at_registration_sha256']=sha(Path(selector.__file__))
        retain(expected,plan)
    res={**load(CODE/'response_resources.json'),'parent_protocol_sha256':sha(expected),'data_parent':sha(CODE/'response_resources.json'),
         'source_maps':entry['maps']}
    retain(wd/'response_resources.json',res)
    return wd,plan

def modules(wd,plan):
    sg.HERE=wd;sg.DIR=wd/'source';sg.PLAN=plan
    responder.HERE=wd;responder.PLAN=plan
    evaluator.HERE=wd;evaluator.PLAN=plan

def snapshot_selector(path):
    payload=path.read_bytes()
    key=hashlib.sha256(payload).hexdigest()
    dest=CODE/'selector_versions'/(key+'.py.gz')
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        if gzip.decompress(dest.read_bytes())!=payload:
            raise ValueError('different selector under the same content address')
    else:
        temp=dest.with_suffix('.part')
        temp.write_bytes(gzip.compress(payload,mtime=0))
        os.replace(temp,dest)
    return key

def select(wd,plan):
    references=load(wd/'source_receipt.json');ledger=wd/'opportunities.jsonl.gz'
    with gzip.open(ledger,'rt') as fp:
        candidates=[json.loads(x) for x in fp]
    jobs={(j['seed'],j['seat'],j['window']):j for j in plan['training_job_registry']}
    choices=[];summary=[];missing=[]
    for g in references['games']:
        s,p=g['seed'],g['seat'];tape=load(ROOT/g['raw_path'])
        for window,(lo,hi) in enumerate(plan['windows']):
            j=jobs[s,p,window];wanted=j['desired_exposure_bits'];m=j['global_map_index']
            prefer=(m+p+window)%2==0
            possible=[r for r in candidates if (r['seed'],r['seat'])==(s,p) and lo<=r['t']<=hi and
                      r['actions']['NONE']['payable_site'] and r['actions']['NONE']['can_pay_now'] and not r['actions']['NONE']['friendly_conflict']]
            def rank(r):
                return (sum(a!=b for a,b in zip(r['stratum_bits'],wanted)),(r['parent_command']=='NONE')!=prefer,
                        selector.ranking(r,m,window)[2])
            selected=None
            for i,r in enumerate(sorted(possible,key=rank)):
                found=selector.route(tape,r)
                if not found:
                    continue
                case=wd/f'case_{s}_{p}_{window}';case.mkdir(exist_ok=True)
                retain(case/'root.json.gz',found)
                selected={'seed':s,'seat':p,'window':window,'step':r['t'],'uid':r['uid'],'cell':r['position'],
                          'price':r['actions']['NONE']['price'],'source':g['raw_path'],'source_sha256':g['raw_sha256'],
                          'case_directory':case.name,'root_sha256':sha(case/'root.json.gz'),'paid_arm':found['paid_arm'],
                          'old':found['old'],'new':found['new'],'source_stratum':list(r['stratum_bits']),
                          'expected_stratum':wanted,'rank_in_window':i+1,'rng_seed':j['rng_seed'],'job_key':j['case_id']}
                choices.append(selected);break
            summary.append({'seed':s,'seat':p,'window':window,'eligible_planter_roots':len(possible),
                            'by_stratum':Counter(''.join(map(str,r['stratum_bits'])) for r in possible),'usable_pair':bool(selected)})
            if selected is None:
                missing.append([s,p,window])
    retain(wd/'selection.json',{'protocol_sha256':sha(wd/'protocol.json'),'source_receipt_sha256':sha(wd/'source_receipt.json'),
                                'all_opportunities_sha256':sha(ledger),'covered_maps':plan['train_maps'],'windows':summary,'cases':choices,
                                'empty_windows':missing,'opponent_future_or_terminal_used_in_selection':False,
                                'selector_sha256':snapshot_selector(Path(__file__)),'shared_decode_sha256':sha(Path(selector.__file__))})

def select_registered(wd,plan):
    # Reconstruct the exact program captured before the source matches, even if main moved.
    key=plan.get('selector_at_registration_sha256')
    if key is None:
        if (wd/'selection.json').is_file():
            return
        raise ValueError('missing pre-source selector registration; inspect original source before selecting')
    if plan['shared_decode_at_registration_sha256']!=sha(Path(selector.__file__)):
        raise ValueError('shared legal-prefix/decoder source has changed')
    archived=CODE/'selector_versions'/(key+'.py.gz')
    payload=gzip.decompress(archived.read_bytes())
    if hashlib.sha256(payload).hexdigest()!=key:
        raise ValueError('corrupt archived root selector')
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'batch_cli.py';path.write_bytes(payload)
        namespace={'__file__':str(path),'__name__':'_frozen_training_selector'}
        exec(compile(payload,str(path),'exec'),namespace)
        namespace['CODE']=CODE
        namespace['select'](wd,plan)
    if load(wd/'selection.json')['selector_sha256']!=key:
        raise ValueError('selection used a different program from registration')

def aggregate(wd):
    examples=[];seen=set();boards=set();provenance=[]
    all_dirs=[*phase_dirs(wd.name),wd]
    for area in all_dirs:
        plan=load(area/'protocol.json');sources=load(area/'source_receipt.json');labels=load(area/'training_labels.json')
        if labels['protocol_sha256']!=sha(area/'protocol.json') or labels['selection_sha256']!=sha(area/'selection.json'):
            raise ValueError('label set, protocol or selection identity mismatch')
        maps={r['seed']:r['map_sha256'] for r in sources['games']}
        if len(set(maps.values()))!=len(maps) or boards&set(maps.values()):
            raise ValueError('source map reused across intakes')
        boards.update(maps.values());provenance.append({'dir':str(area.relative_to(CODE)),'labels_sha256':sha(area/'training_labels.json')})
        for r in labels['pairs']:
            key=(plan['parent_pack_sha256'],maps[r['seed']],r['seat'],r['t'],r['focal_uid'])
            path=area/r['features'];label_path=path.parent/'training_label.json'
            if key in seen or sha(path)!=r['root_sha256'] or load(label_path)!=r:
                raise ValueError('duplicate or different state / label evidence')
            seen.add(key)
            examples.append({'key':list(key),'source_intake':str(area.relative_to(CODE)),'features':str(path.relative_to(CODE)),
                             'label':str(label_path.relative_to(CODE)),'label_sha256':sha(label_path),'filled':r['pay_intent_filled'],
                             'target':r['target_delta_official_margin'],'seed':r['seed'],'seat':r['seat'],'observed_stratum':r['observed_stratum']})
    filled=[r for r in examples if r['filled']]
    positive=sorted({r['seed'] for r in filled if r['target']>0});negative=sorted({r['seed'] for r in filled if r['target']<0})
    complete=len(boards)>=8 and len(filled)>=32 and len(positive)>=2 and len(negative)>=2
    doc={'policy_sha256':plan['parent_pack_sha256'],'source_intakes':provenance,'distinct_maps':len(boards),
         'complete_pairs':len(examples),'filled_pairs':len(filled),'positive_maps':positive,'negative_maps':negative,
         'fit_support_gate':complete,'held_out_games':0,'fit_performed':False,'examples':examples}
    retain(wd/'aggregate.json',doc)
    return doc

def run_pairs(wd,plan):
    # One child process per arm is mandatory: original Master3s2 has process-global state.
    items=responder.records()
    def lane(lane_no,cpus):
        for index in range(lane_no,len(items),2):
            d=wd/items[index]['case_directory']
            stamp=d/'resource_at_start.json'
            jobs=other_native(wd)
            while jobs:
                time.sleep(8);jobs=other_native(wd)
            if not stamp.exists():
                retain(stamp,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpus':cpus,'original_native_elsewhere':jobs,
                              'cgroup_cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                              'cgroup_stat_at_launch':Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines(),
                              'loadavg_at_launch':Path('/proc/loadavg').read_text().strip(),'allocation_sha256':sha(wd/'response_resources.json')})
            def one(worker):
                k,cpu=worker;arm='parent' if (k+index//2)%2==0 else 'alternative'
                subprocess.run(['taskset','-c',str(cpu),sys.executable,'-u',str(Path(__file__)),wd.name,'arm','--index',str(index),'--arm',arm],env=ENV,check=True)
            with ThreadPoolExecutor(max_workers=2) as arms:
                list(arms.map(one,enumerate(cpus)))
    with ThreadPoolExecutor(max_workers=2) as lanes:
        list(lanes.map(lambda x:lane(*x),enumerate(load(wd/'response_resources.json')['paired_case_lanes'])))

def other_native(wd):
    jobs=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try: args=[os.fsdecode(q) for q in (proc/'cmdline').read_bytes().split(b'\0') if q]
        except OSError:continue
        if len(args)<2 or 'python' not in Path(args[0]).name:continue
        if any(q.endswith('batch_cli.py') for q in args) and wd.name in args:continue
        scripts=[q for q in args[1:] if q.endswith('.py')]
        if any(q.endswith('tree_pack_arena.py') or ('tree_response_training/pair_games.py' in q and 'play' in args)
               or '/workforce_policy/branch_' in q or '/opponent_transfer/run.py' in q for q in scripts):jobs.append(int(proc.name))
    return jobs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('batch');ap.add_argument('mode',choices=('register','run','arm','aggregate'));ap.add_argument('--index',type=int);ap.add_argument('--arm',choices=('parent','alternative'))
    a=ap.parse_args();wd,plan=register(a.batch);modules(wd,plan)
    if a.mode=='register':
        print(wd/'protocol.json',flush=True);return
    if a.mode=='arm':
        if a.index is None or a.arm is None:raise ValueError('missing case/arm')
        case_dir=wd/responder.records()[a.index]['case_directory']
        with (case_dir/f'.response_{a.arm}.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            responder.play(a.index,a.arm)
        return
    if a.mode=='aggregate':
        doc=aggregate(wd);write_report(wd)
        print(doc['complete_pairs']);return
    with (wd/'.intake.schedule.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (wd/'clearance.json').is_file():
            subprocess.run(['taskset','-c','4',sys.executable,'-u',str(CODE/'batch_clearance.py'),str(wd/'protocol.json')],check=True,env=ENV)
        sg.check_maps()
        if not (wd/'source_receipt.json').is_file():
            while other_native(wd):time.sleep(8)
            sg.play()
        else:sg.verify()
        if not (wd/'opportunities.jsonl.gz').is_file():
            opportunities.save(wd/'opportunities.jsonl.gz',intake_dir=wd)
        if not (wd/'selection.json').is_file():select_registered(wd,plan)
        run_pairs(wd,plan)
        evaluator.finish()
        doc=aggregate(wd)
        write_report(wd)
        print({k:doc[k] for k in ('distinct_maps','complete_pairs','filled_pairs','fit_support_gate')},flush=True)

if __name__=='__main__':
    main()
