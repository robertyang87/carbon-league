"""Match the original search work on the early held-out roots."""
import argparse
import importlib.util

from build import build
from train import FAMILY, HERE, LATER, PARENT, ROOT, select, sha

path = FAMILY/'native_critic_update/public_search/check.py'
spec = importlib.util.spec_from_file_location('fixed_original_search_readout', path)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
STEPS = (45, 85)
NATIVE_PARENT = FAMILY/'ladder/prediction_feedback/package/candidate.py'


def registration():
    rows = sorted((t for t in select() if t['panel'] in LATER), key=lambda t: (t['game']['seed'], t['game']['seat']))
    pick = sorted({r['game']['seed'] for r in rows})[::2]
    tickets = []
    for item in rows:
        g = item['game']
        seed, seat = g['seed'], g['seat']
        if seed not in pick:
            continue
        for step in STEPS:
            rng = int(sha(f'live-tree-income/1/{seed}/{seat}/{step}'.encode())[:8], 16)
            tickets.append(dict(archive=str((ROOT/item['archive']).relative_to(FAMILY)),
                                archive_sha256=item['archive_sha256'], arm=g['arm'], seed=seed, seat=seat,
                                terrain=g['terrain'], registration_sha256=item['registration_sha256'], step=step, rng=rng))
    assert len(tickets) == 16
    return dict(schema='native-existing-tree-leaf/1', packages=shared.DIGESTS,
                selection='Take alternating seed numbers over the two 4-map held panels, both seats, t45 and t85.',
                correction='One quarter of the model-versus-duration-only cash tail, scaled by 3000 and capped at 0.06 value units.',
                tickets=tickets)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--register', action='store_true')
    cli.add_argument('--cpu', type=int)
    args = cli.parse_args()
    shared.HERE, shared.PARENT, shared.CANDIDATE = HERE, NATIVE_PARENT, HERE/'candidate.py'
    shared.DIGESTS = (PARENT, build())
    shared.STEPS, shared.registration = STEPS, registration
    if args.register:
        locked = shared.encode(registration())
        plan = HERE/'search_protocol.json'
        assert not plan.exists() or plan.read_bytes() == locked
        plan.write_bytes(locked)
        print(sha(locked))
    else:
        # The shared runner reads protocol.json and writes results.json.gz.
        # Give it a separate directory so the frozen training recipe stays untouched.
        shared.HERE = HERE/'search'
        shared.HERE.mkdir(exist_ok=True)
        dest = shared.HERE/'protocol.json'
        lock = (HERE/'search_protocol.json').read_bytes()
        assert not dest.exists() or dest.read_bytes() == lock
        dest.write_bytes(lock)
        shared.execute(args.cpu)
