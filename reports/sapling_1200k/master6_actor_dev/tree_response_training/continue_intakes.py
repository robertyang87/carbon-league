"""Single-owner continuation for existing registered training tasks; never opens development or formal matches."""
import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from batch_cli import CODE,ENV,load,sha


def active_lock(path):
    if not path.is_file():
        return False
    with path.open('a') as fd:
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def verify_sealed(area):
    labels=load(area/'training_labels.json')
    protocol=load(area/'protocol.json')
    sources=load(area/'source_receipt.json')
    sel=load(area/'selection.json')
    if (labels['protocol_sha256']!=sha(area/'protocol.json') or labels['selection_sha256']!=sha(area/'selection.json')
            or sources['official_sha256']!=sha(area/'source/official_replay.json')
            or sources['protocol_sha256']!=sha(area/'protocol.json') or sel['source_receipt_sha256']!=sha(area/'source_receipt.json')
            or sel['all_opportunities_sha256']!=sha(area/'opportunities.jsonl.gz') or len(sel['cases'])!=len(labels['pairs'])):
        raise ValueError(f'unmatched summary/official file or incomplete case set in {area}')
    selector_key=sel['selector_sha256']
    if area==CODE:
        if selector_key!=sha(CODE/'select_roots.py'):
            raise ValueError('initial root selector source differs')
    else:
        archived=CODE/'selector_versions'/(selector_key+'.py.gz')
        if hashlib.sha256(gzip.decompress(archived.read_bytes())).hexdigest()!=selector_key:
            raise ValueError('registered selector archive was modified')
    if (sources['clearance_sha256']!=sha(area/'clearance.json')
            or sel['protocol_sha256']!=sha(area/'protocol.json')):
        raise ValueError('different clearance or upstream selection contract')
    expected={(s,seat) for s in protocol['train_maps'] for seat in (0,1)}
    if len(sources['games'])!=len(expected) or {(r['seed'],r['seat']) for r in sources['games']}!=expected:
        raise ValueError('incomplete or repeated source map and physical seat')
    board_by_seed={}
    for r in sources['games']:
        from source_games import ROOT
        if sha(ROOT/r['raw_path'])!=r['raw_sha256'] or board_by_seed.setdefault(r['seed'],r['map_sha256'])!=r['map_sha256']:
            raise ValueError('source raw tape or opening differs')
    registered={(r['seed'],r['seat'],r['step'],r['uid']):r for r in sel['cases']}
    seen=set()
    for r in labels['pairs']:
        key=r['seed'],r['seat'],r['t'],r['focal_uid']
        rec=registered[key]
        path=area/r['features']
        if key in seen or sha(path)!=rec['root_sha256'] or load(path.parent/'training_label.json')!=r:
            raise ValueError('duplicate or corrupt training labels')
        seen.add(key)
        for side in ('pay','defer'):
            v=r[side];case=path.parent
            native=load(case/f'live_{v["native_arm"]}.json')
            if (v['actual_action_tape_sha256']!=sha(case/native['action_tape'])
                    or v['shared_stream_sha256']!=sha(case/'live_shared_init.json') or native['score']!=v['scores_by_physical_seat']):
                raise ValueError('changed paired trajectory or terminal label')
    return labels['pairs'],board_by_seed


def board_state():
    q=load(CODE/'next_intake.json')
    if q['source_data_sha256']!=sha(CODE/'training_labels.json'):
        raise ValueError('unbound continuation plan')
    all_rows,seen=verify_sealed(CODE)
    rows=list(all_rows);boards=set(seen.values());finished=[]
    initial_selection=load(CODE/'selection.json')
    expected_windows=len(initial_selection['windows']);empty_windows=list(initial_selection['empty_windows'])
    pending=None
    for reg in q['batches']:
        filled=[r for r in rows if r['pay_intent_filled']]
        if reg['activation'].startswith('fill_only') and len(boards)>=8 and len(filled)>=32:
            continue
        area=CODE/'batches'/reg['name']
        if (area/'aggregate.json').exists():
            added,new=verify_sealed(area)
            selection=load(area/'selection.json')
            expected_windows+=len(selection['windows']);empty_windows.extend(selection['empty_windows'])
            if boards&set(new.values()) or len(new)!=len(reg['maps']):
                raise ValueError('source terrain reused')
            chain=load(area/'aggregate.json')
            if chain['source_intakes'][-1]['labels_sha256']!=sha(area/'training_labels.json'):
                raise ValueError('aggregate did not consume its own labels')
            boards.update(new.values());rows.extend(added);finished.append(reg['name'])
            if chain['distinct_maps']!=len(boards) or chain['complete_pairs']!=len(rows):
                raise ValueError('incorrect across-map aggregate')
            continue
        source_dir=area/'source'
        source_n=len(load(source_dir/'results.json')['games']) if (source_dir/'results.json').is_file() else 0
        if not (area/'clearance.json').exists():stage='clearance'
        elif not (area/'source_receipt.json').exists():stage='source_games'
        elif not (area/'opportunities.jsonl.gz').exists():stage='public_opportunities'
        elif not (area/'selection.json').exists():stage='root_selection'
        elif not (area/'training_labels.json').exists():stage='paired_responses'
        else:stage='aggregate'
        pending={'batch':reg['name'],'stage':stage,'registered':(area/'protocol.json').is_file(),'source_games_completed':source_n,
                 'source_games_registered':reg['expected_source_games'],'paired_arms_completed':sum(1 for a in ('parent','alternative') for _ in area.glob(f'case_*/live_{a}.json')),
                 'live_batch_lock':active_lock(area/'.intake.schedule.lock')}
        break
    complete=[r for r in rows if r['pay_intent_filled']]
    positive=sorted({r['seed'] for r in complete if r['target_delta_official_margin']>0})
    negative=sorted({r['seed'] for r in complete if r['target_delta_official_margin']<0})
    support=len(boards)>=8 and len(complete)>=32 and len(positive)>=2 and len(negative)>=2
    terminal='collecting' if pending else ('support_met' if support else 'registered_fill_exhausted_support_incomplete')
    missing={'distinct_training_maps':max(0,8-len(boards)),'actual_paid_pairs':max(0,32-len(complete)),
             'positive_effect_maps':max(0,2-len(positive)),'negative_effect_maps':max(0,2-len(negative))}
    return {'terminal_state':terminal,'finished_batches':finished,'verified_planned_windows':expected_windows,
            'missing_root_windows':empty_windows,'remaining_training_support':missing,'verified_distinct_maps':len(boards),'verified_paired_roots':len(rows),
            'verified_actual_payments':len(complete),'positive_maps':positive,'negative_maps':negative,
            'fit_support':support,
            'pending':pending,'reserved_development_maps':q['protected_unopened_development_maps'],'development_scheduled':False}


def supervise():
    with (CODE/'.continuation.lock').open('a') as fd:
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('continuation already supervised',flush=True);return
        announced=None
        while True:
            state=board_state()
            phase=state['pending']
            signature=(phase['batch'],phase['stage']) if phase else ('complete',str(state['fit_support']))
            if signature!=announced:
                print(json.dumps(state,ensure_ascii=False),flush=True);announced=signature
            if phase is None:
                return
            if phase['live_batch_lock']:
                time.sleep(20);continue
            # Another explicit owner may start between status and launch; the per-batch OS lock is authoritative.
            cmd=[sys.executable,'-u',str(CODE/'batch_cli.py'),phase['batch'],'run']
            run=subprocess.run(cmd,env=ENV)
            if run.returncode:
                if active_lock(CODE/'batches'/phase['batch']/'.intake.schedule.lock'):
                    time.sleep(15);continue
                raise RuntimeError(f'{phase["batch"]} interrupted with code {run.returncode}; keep all receipts and resume after diagnosis')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('status','drain'));a=ap.parse_args()
    if a.mode=='status':
        x=board_state();x['live_global_lock']=active_lock(CODE/'.continuation.lock');print(json.dumps(x,ensure_ascii=False,indent=2))
    else:
        supervise()
