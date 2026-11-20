"""Offer two single-worker variations before stochastic teams within the same search budget."""
import argparse
import importlib.util
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
SOURCE = FAMILY/'ladder/prediction_feedback/package/candidate.py'
TARGET = HERE/'candidate.py'
PARENT_SHA = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'
BEFORE = b'_S_GEN = ((2.5, 8), (1.0, 3), (3.5, 8), (0.0, 6), (5.0, 8))'
AFTER = b'_S_GEN = ((0.0, 2), (2.5, 8), (1.0, 3), (3.5, 8), (0.0, 6), (5.0, 8))'
spec = importlib.util.spec_from_file_location('verified_public_search', FAMILY/'native_critic_update/public_search/check.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def build():
    source = SOURCE.read_bytes()
    assert check.sha(source) == PARENT_SHA and source.count(BEFORE) == 1
    data = source.replace(BEFORE, AFTER)
    if TARGET.exists():
        assert TARGET.read_bytes() == data
    else:
        TARGET.write_bytes(data)
    return check.sha(data)


def registration():
    tickets = []
    for path in sorted((FAMILY/'v4_replication_dev/hosted').glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            raw = archive.read('parent/registration.json')
        reg = json.loads(raw)
        assert reg['files']['candidate']['sha256'] == PARENT_SHA
        assert reg['files']['opponent']['sha256'] == check.ORIGINAL
        seed, seat = reg['seed'], reg['candidate_seat']
        assert seed in range(621758610,621758614)
        for turn in check.STEPS:
            rng = int(check.sha(f'single-worker-priority/1/{seed}/{seat}/{turn}'.encode())[:8],16)
            tickets.append(dict(archive=str(path.relative_to(FAMILY)),archive_sha256=check.sha(path.read_bytes()),
                                arm='parent',registration_sha256=check.sha(raw),seed=seed,seat=seat,
                                terrain=reg['physical_map_sha256'],step=turn,rng=rng))
    assert len(tickets) == len({(r['seed'], r['seat'], r['step']) for r in tickets}) == 16
    return dict(schema='single-worker-priority/1',packages=check.DIGESTS,tickets=tickets,
                design='Two deterministic single-worker alternatives first, preserve the original menu generators for remaining places, unchanged clock and native score. ABBA natural search and matched-work controls on all 16 public roots.')


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    args = cli.parse_args()
    check.HERE,check.PARENT,check.CANDIDATE = HERE,SOURCE,TARGET
    check.DIGESTS = (PARENT_SHA,build())
    check.registration = registration
    if args.register:
        planned = check.encode(registration())
        path = HERE/'protocol.json'
        assert not path.exists() or path.read_bytes()==planned
        path.write_bytes(planned)
        print(check.DIGESTS[1],check.sha(planned))
    else:
        check.execute(args.cpu)
