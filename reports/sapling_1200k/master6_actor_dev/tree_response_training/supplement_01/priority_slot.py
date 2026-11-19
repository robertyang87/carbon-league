"""A single shared original-opponent slot between the sealed original fifth and supplementary sixth intake."""
import argparse
from contextlib import contextmanager
import fcntl
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent
sys.path.insert(0,str(TRAIN))
from select_roots import load,retain,sha
from source_games import ROOT,gate
from continue_intakes import verify_sealed

SLOT=HERE/'priority_dev'
EXPECTED=HERE/'batches/intake_06'
RESERVATION=SLOT/'reservation.json'


def check(ok,msg):
    if not ok:raise ValueError(msg)


def allocated():
    d=load(RESERVATION)
    closed=TRAIN/d['after']
    check(sha(closed/'aggregate.json')==d['after_aggregate_sha256'] and not (EXPECTED/'source/results.json').exists(),
          'outside the reserved boundary or different outgoing training seal')
    return d


def grant():
    with (SLOT/'.state.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        allocation=allocated()
        proof=SLOT/'granted.json'
        card={'reservation_sha256':sha(RESERVATION),'training_05_aggregate_sha256':allocation['after_aggregate_sha256'],
              'training_06_source_games_at_transfer':0,'native_cpus':[0,1],'study':allocation['study']}
        if not proof.exists():verify_sealed(TRAIN/allocation['after'])
        retain(proof,card)
        return proof


@contextmanager
def native_session():
    with (SLOT/'.native-owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        grant()
        check(not any((SLOT/name).exists() for name in ('released.json','skipped.json')),'slot already returned')
        yield


def wait_before_source(wd,plan):
    if wd.resolve()==HERE/'batches/intake_07':
        from early_lease import wait_before_source as wait_for_early_lane
        wait_for_early_lane(wd,plan)
        return
    if wd.resolve()!=EXPECTED:return
    check(plan['intake_name']==EXPECTED.name,'different training block')
    while True:
        if (SLOT/'released.json').exists():
            from priority_verify import check_release
            check_release();return
        if (SLOT/'skipped.json').exists():
            from priority_verify import check_skip
            check_skip();return
        grant()
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('grant','release','skip','verify'))
    a=p.parse_args()
    if a.command=='grant':print(grant())
    elif a.command=='release':
        from priority_verify import release
        print(release())
    elif a.command=='skip':
        from priority_verify import skip
        print(skip())
    else:
        from priority_verify import check_release,check_skip
        print(check_release() if (SLOT/'released.json').exists() else check_skip())
