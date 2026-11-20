"""Freeze already exposed terrains, root times and the offline search teacher."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PARENT = BASE / 'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'
TRAIN = (590, 591, 592, 593, 594, 595, 598, 599, 600, 601, 602, 603)
HELD = (604, 605, 606, 607)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    raw = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f'Existing frozen evidence differs: {path}')
    path.write_bytes(raw)


def register():
    assert sha(PARENT.read_bytes()) == PARENT_SHA
    sources = {}
    for folder in sorted(BASE.rglob('hosted')):
        for path in sorted(folder.glob('*.zip')):
            with zipfile.ZipFile(path) as archive:
                for role in ('parent', 'cold_repair'):
                    entry = f'{role}/registration.json'
                    if entry not in archive.namelist():
                        continue
                    row = json.loads(archive.read(entry))
                    seed = row['seed']
                    short = seed - 621758000
                    if short not in TRAIN + HELD or row['candidate_seat'] != short % 2:
                        continue
                    if seed in sources:
                        raise ValueError(f'Duplicate source: {seed}')
                    sources[seed] = dict(seed=seed, seat=row['candidate_seat'], terrain=row['physical_map_sha256'],
                                         archive=str(path.relative_to(BASE)), sha256=sha(path.read_bytes()),
                                         role=role, split='train' if short in TRAIN else 'held')
    assert len(sources) == len({v['terrain'] for v in sources.values()}) == len(TRAIN + HELD)
    save(HERE/'protocol.json', dict(parent_sha256=PARENT_SHA, sources=[sources[s] for s in sorted(sources)],
         steps=[48, 100, 160, 224, 291], cost='offline gold: 8 candidates, 8 joint transitions, no early pruning or clock; A0 always reserved',
         search='same native V2 root, same opponent root command. Train continuation seed and held evaluation seed 0; additional independent held continuation 79031 changes later actions only.',
         target='Deep value for every candidate at depth 8 (exact simulated expiry for step 291); fit centered H8 minus H1 using frozen H1 critic activations and map-equal ridge. Training trajectories are archived play; teacher labels are modeled responses.',
         gate='Both held future draws must lower map-equal final H8 regret after H1 half then unchanged H4 quarter and preserve A0, improve at least 3/4 terrains, and do no worse on terminal roots. Only then eligible for independent true responses; otherwise no deployment.',
         tie='native menu order. Always retain A0. Within a root compare all methods against same full teacher menu for that future draw.',
         ridge_penalty=0.1, final_market='none; 608,609 and the formal maps excluded'))
if __name__ == '__main__':
    register()
