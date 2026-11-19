"""Rebuild the Sapling local-map exclusions from committed game records; never draw exam maps."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from map_identity import local_board

BASE = 'reports/sapling_1200k'
ORIGINAL_SEED_FILE = BASE + '/master6_actor_dev/data_prior_local_seeds.json'
ARCHIVES = ['reports/sapling_exam_results.tgz', 'reports/sapling_exam_r2_results.tgz',
            'reports/sapling_exam_v2_200_results.tgz']


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def build(revision):
    commit = git('rev-parse', '--verify', revision + '^{commit}').decode().strip()
    paths = git('ls-tree', '-r', '--name-only', commit, BASE).decode().splitlines()
    results = sorted(p for p in paths if p.endswith('/results.json'))
    seen = set()
    witnessed = {}
    sources = []
    for path in [*results, ORIGINAL_SEED_FILE, *ARCHIVES]:
        raw = git('show', commit + ':' + path)
        if path in ARCHIVES:
            rows = []
            with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
                for member in archive:
                    if member.isfile() and '/bigeval_live/' in member.name and member.name.endswith('.jsonl'):
                        rows.extend(json.loads(line) for line in archive.extractfile(member) if line.strip())
            if not rows:
                raise ValueError(f'No per-game seat records in {path}')
        elif path == ORIGINAL_SEED_FILE:
            rows = [{'seed': s} for s in json.loads(raw)['seeds']]
        else:
            document = json.loads(raw)
            rows = document.get('games', []) if isinstance(document, dict) else []
        if isinstance(rows, dict):
            rows = [row for arm in rows.values() for row in arm]
        seeds = set()
        for row in rows:
            if not isinstance(row, dict) or type(row.get('seed')) is not int:
                raise ValueError(f'Invalid game record in {path}')
            seed = row['seed']
            seeds.add(seed)
            terrain = row.get('initial_terrain_u8_sha256')
            if terrain is not None and witnessed.setdefault(seed, terrain) != terrain:
                raise ValueError(f'Conflicting observed terrain for seed {seed}')
        sources.append(dict(path=path, sha256=hashlib.sha256(raw).hexdigest(), rows=len(rows), seeds=sorted(seeds)))
        seen.update(seeds)
    maps = []
    for seed in sorted(seen):
        terrain = local_board(seed)
        if seed in witnessed and witnessed[seed] != terrain:
            raise ValueError(f'Generated map and observed game record disagree for seed {seed}')
        maps.append(dict(seed=seed, initial_terrain_u8_sha256=terrain, witnessed=seed in witnessed))
    return dict(revision=commit, sources=sources, maps=maps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision')
    parser.add_argument('--out', type=Path, default=HERE/'exposure.json')
    args = parser.parse_args()
    document = build(args.revision)
    args.out.write_text(json.dumps(document, separators=(',', ':')) + '\n')
    print(len(document['maps']), 'local maps,', sum(m['witnessed'] for m in document['maps']), 'with archived terrain hash')
