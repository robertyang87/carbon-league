"""Replay retained action pairs against the official referee, without loading either agent."""
import argparse
import gzip
import json
import tempfile
import zipfile
from pathlib import Path
from paired import HERE, D, LOCK, module, sha


def verify(path, ticket, recovery):
    record = json.loads(gzip.decompress(path.read_bytes()))
    src = D / ticket['source']
    assert sha(src.read_bytes()) == ticket['zip_sha256']
    assert record['protocol'] == LOCK
    with zipfile.ZipFile(src) as archive, tempfile.TemporaryDirectory() as workdir:
        reg = Path(workdir) / 'registration.json'
        reg.write_bytes(archive.read('parent/registration.json'))
        recovery.live.configure(reg, Path(workdir) / 'claim', Path(workdir) / 'lock')
        env, _ = recovery.live.official()
    bank = [float(recovery.live.RULE['overage_seconds_per_player'])] * 2
    for expected_step, row in enumerate(record['rows']):
        assert not env.done and expected_step == row['step'] == env.states[0].observation.step
        for side in (0, 1):
            obs = recovery.freight.public(env, side, bank)
            assert recovery.freight.digest(obs) == row['public_sha256'][side], (path.name, expected_step, side)
        bank = [value - max(0., clock['wall'] - 1.) for value, clock in zip(bank, row['clock'])]
        env.step(row['orders'], [])
        assert [p[0] for p in env.states[0].observation.players] == row['cash']
    assert env.done
    assert [float(p.reward) for p in env.states] == record['result']['official_scores']
    assert [p.status for p in env.states] == record['result']['statuses']
    assert bank == record['result']['final_banks']
    return dict(sha256=sha(path.read_bytes()), steps=len(record['rows']), verified_seat_observations=2*len(record['rows']),
                official_scores=[float(p.reward) for p in env.states], statuses=[p.status for p in env.states])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', nargs='*', help='local arm files for a partial verification; default is all four')
    args=parser.parse_args()
    plan=(HERE/'protocol.json').read_bytes(); assert sha(plan)==LOCK
    tickets=json.loads(plan)['tickets']
    recovery=module('opening_replay_recovery', D/'endgame_cash_projection/response_check/recovery.py')
    result={}
    known={f"{t['seed']}_{arm}.json.gz":t for t in tickets for arm in t['run_order']}
    for name in args.arms or list(known):
        result[name]=verify(HERE/name, known[name], recovery)
    print(json.dumps(result,indent=2,sort_keys=True))
    if not args.arms:
        (HERE/'replay_check.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
