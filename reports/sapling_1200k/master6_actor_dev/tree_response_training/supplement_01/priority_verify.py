"""Verify the eight native games independently before returning the original-opponent slot."""
import fcntl
import importlib.util
import json
from pathlib import Path

from priority_slot import TRAIN,SLOT,RESERVATION,sha,load,retain,check

DEV=TRAIN.parent/'symmetry_dev'
REGISTRATION='55ae16a63cf167aef015bc2f10a8073ef26c13287ad18480c6402b86b235aaef'


def module():
    spec=importlib.util.spec_from_file_location('four_view_receipt',DEV/'run.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def shared():
    g=load(SLOT/'granted.json');book=load(RESERVATION)
    check(g['study']==book['study']=='v1_four_view_policy_ensemble'
          and g['reservation_sha256']==sha(RESERVATION) and g['training_05_aggregate_sha256']==book['after_aggregate_sha256']
          and sha(TRAIN/book['after']/'aggregate.json')==book['after_aggregate_sha256']
          and sha(DEV/'protocol.json')==REGISTRATION,'wrong slot, previous aggregate or development registration')
    return {'grant_sha256':sha(SLOT/'granted.json'),'registration_sha256':REGISTRATION}


def compute(full):
    common=shared();m=module()
    plan,release,clear=m.approved()
    check(plan['parent_pack_sha256']==load(RESERVATION)['parent_pack_sha256'] and plan['examiner_md5']==load(RESERVATION)['examiner_md5'],
          'wrong parent or original opponent')
    reference=json.loads(json.dumps(m.certified(official_again=full)))
    check(reference==load(DEV/'comparison.json'),'incorrect paired official development result')
    return {**common,'release_identity_sha256':sha(DEV/'release.json'),'mechanism_sha256':sha(DEV/'mechanism.json'),
            'comparison_sha256':sha(DEV/'comparison.json'),'arena_binding_sha256':sha(DEV/'arena_binding.json'),
            'signed_native_games':len(reference['paired_map_seats'])*2,'source_scores':reference['competitors'],
            'verdict':reference['verdict']}


def check_release():
    receipt=load(SLOT/'released.json')
    check(not (SLOT/'skipped.json').exists() and receipt==compute(False),'development release differs')
    return receipt


def release():
    with (SLOT/'.state.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (SLOT/'released.json').exists():return check_release()
        check(not (SLOT/'skipped.json').exists(),'slot already closed for an unplayed failure')
        receipt=compute(True)
        check(receipt['signed_native_games']==8,'incomplete source games')
        retain(SLOT/'released.json',receipt)
        return receipt


def skipped():
    common=shared()
    for arm in ('parent','candidate'):
        directory=DEV/arm
        check(not list(directory.glob('raw_*')) and
              (not (directory/'results.json').exists() or not load(directory/'results.json')['games']),
              'cannot skip after a native trajectory has been played')
    mechanism=load(DEV/'mechanism_skip.json');receipt=load(DEV/'mechanism.json')
    check(receipt.get('passed') is False and receipt['protocol_sha256']==REGISTRATION
          and mechanism['protocol_sha256']==REGISTRATION and mechanism['mechanism_sha256']==sha(DEV/'mechanism.json'),
          'no signed, preregistered mechanism failure')
    return {**common,'signed_native_games':0,'mechanism_sha256':sha(DEV/'mechanism.json'),'reason_sha256':sha(DEV/'mechanism_skip.json')}


def check_skip():
    receipt=load(SLOT/'skipped.json')
    check(not (SLOT/'released.json').exists() and receipt==skipped(),'invalid skip receipt')
    return receipt


def skip():
    proof=skipped()
    check(not (SLOT/'released.json').exists(),'completed slot cannot be skipped')
    retain(SLOT/'skipped.json',proof)
    return proof
