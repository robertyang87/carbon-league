"""Restore the public requests from both seats and verify each against the live transport digest."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import matrix


def recover(folder):
    live = matrix.play.live
    receipt = matrix.read(folder / 'receipt.json')
    reg = folder / 'registration.json'
    live.configure(reg, folder / 'attempt.json', folder.parent / '.match.lock')
    env, _ = live.official()
    tape = folder / 'turns.jsonl'
    if tape.exists() and matrix.sha(tape) != receipt['turns_sha256']:
        raise ValueError("Transport archive changed")
    records = []
    bank = [5., 5.]
    if tape.is_file():
        for n, line in enumerate(tape.read_bytes().splitlines()):
            r=json.loads(line)
            if env.done or r['step'] != n or env.states[0].observation.step != n:
                raise ValueError("Invalid replay clock")
            observations=[live.observe(env,s,bank[s]) for s in (0,1)]
            fingerprints=[hashlib.sha256(live.encoded(v)).hexdigest() for v in observations]
            if fingerprints != r['public_observation_sha256']:
                raise ValueError("Recorded public view diverged")
            records.append(live.encoded(dict(step=n, observations=observations, public_observation_sha256=fingerprints)))
            bank = r['overage_remaining_seconds']
            if not live.failures(r['calls']):
                env.step([v['response']['action'] for v in r['calls']])
    if env.states[0].observation.step != receipt['official_transitions']:
        raise ValueError("Physical move count differs from receipt")
    if receipt.get('status')=='complete' and [s.reward for s in env.states] != receipt['official_scores_by_seat']:
        raise ValueError("Official end differs")
    return gzip.compress(b''.join(records),mtime=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args=parser.parse_args();raw=recover(args.folder)
    dest=args.folder/'observations.jsonl.gz'
    if dest.exists() and dest.read_bytes()!=raw:raise ValueError("Existing observations differ")
    dest.write_bytes(raw)
    print(len(gzip.decompress(raw).splitlines()), hashlib.sha256(raw).hexdigest())
