"""Count distinct complete v1 contests against the full original at a frozen commit."""
from collections import Counter
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import statistics

from phase_census import PACK, EXAM, git

ORIGINAL_SHA256 = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'


def parent_tapes(commit, path, doc):
    directory = Path(path).parent
    header = doc.get('header', {})
    if header.get('pack_sha256') == PACK and header.get('examiner_md5') == EXAM:
        if doc.get('status') != 'complete':
            raise ValueError(f'Unfinished records at {path}')
        for row in doc['games']:
            yield row, str(directory / row['raw_file'])
        return

    binding = doc.get('binding', {})
    if binding.get('parent_sha256') != PACK or binding.get('original_sha256') != ORIGINAL_SHA256:
        return
    if doc.get('status') != 'complete':
        raise ValueError(f'Unfinished mixed-arm records at {path}')
    for result in doc['games']:
        if result['arm'] != 'parent':
            continue
        seed, seat = result['seed'], result['seat']
        parent_dir = directory / 'parent'
        receipt = json.loads(git('show', f'{commit}:{parent_dir}/receipt_{seed}_{seat}.json'))
        row, official = receipt['game'], receipt['official']
        if (receipt['binding'] != binding or (seed, seat) != (row['seed'], row['candidate_seat'])
                or row['raw_sha256'] != result['raw_sha256'] or official['transcript_sha256'] != row['raw_sha256']
                or official['statuses'] != ['DONE', 'DONE'] or official['official_reward_by_seat'] != result['scores']
                or row['official_score_by_seat'] != result['scores'] or row['margin'] != result['margin']
                or official['result'] != result['result']):
            raise ValueError(f'Conflicting mixed-arm result for {seed}, seat {seat} at {path}')
        yield row, str(parent_dir / row['raw_file'])


def collect(revision):
    commit = git('rev-parse', '--verify', revision + '^{commit}').decode().strip()
    paths = git('ls-tree', '--name-only', '-r', commit, 'reports/sapling_1200k').decode().splitlines()
    games = {}
    for path in paths:
        if not path.endswith('/results.json'):
            continue
        doc = json.loads(git('show', commit + ':' + path))
        for row, tape_path in parent_tapes(commit, path, doc):
            packed = git('show', commit + ':' + tape_path)
            tape = json.loads(gzip.decompress(packed))
            seat = row['candidate_seat']
            score = tape['official_score_by_seat']
            margin = round(score[seat] - score[1-seat], 2)
            ident = hashlib.sha256(bytes(map(int, tape['initial_full_carbon']))).hexdigest()
            actions_sha = hashlib.sha256(json.dumps(tape['actions'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            cfg = tape['configuration']
            if (not row['valid'] or tape['errors'] or row['audit_messages']
                    or hashlib.sha256(packed).hexdigest() != row['raw_sha256']
                    or tape['pack_sha256'] != PACK or tape['examiner_md5'] != EXAM
                    or not cfg['_S_ENGINE'] or cfg['_TEMP'] != 1 or cfg['_SEED'] is not None
                    or row['search']['searched'] <= 0 or len(tape['actions']) != row['actual_transitions']
                    or actions_sha != row['actions_sha256']
                    or tape['candidate_seat'] != seat or row['seed'] != tape['seed']
                    or ident != tape['initial_terrain_u8_sha256'] or ident != row['initial_terrain_u8_sha256']
                    or score != row['official_score_by_seat'] or margin != round(row['margin'],2)):
                raise ValueError(f'Inconsistent signed contest at {tape_path}')
            key = (ident, seat, actions_sha)
            if key not in games:
                games[key] = dict(seed=row['seed'], seat=seat, map=ident, margin=margin,
                                  steps=row['actual_transitions'], searched=row['search']['searched'],
                                  actions_sha256=actions_sha, raw_sha256=row['raw_sha256'], sources=[])
            if games[key]['margin'] != margin:
                raise ValueError(f'Conflicting duplicate score at {tape_path}')
            games[key]['sources'].append(tape_path)
    rows = sorted(games.values(), key=lambda r: (r['seed'], r['seat']))
    if not rows or set(Counter((r['map'],r['seat']) for r in rows).values()) != {1}:
        raise ValueError('New replicates require an explicit map/seat weighting policy')
    if any({r['seat'] for r in rows if r['map'] == m} != {0,1} for m in {r['map'] for r in rows}):
        raise ValueError('Missing a seat')

    def summary(batch):
        margins = [r['margin'] for r in batch]
        return dict(games=len(batch), wins=sum(x>0 for x in margins), draws=margins.count(0),
                    mean_margin=round(statistics.mean(margins),3), closest_margin=max(margins),
                    full_length=sum(r['steps']==299 for r in batch))

    return dict(revision=commit, parent_sha256=PACK, examiner_md5=EXAM, scope='archived development/training source maps',
                maps=len({r['map'] for r in rows}), all=summary(rows),
                by_seat=[summary([r for r in rows if r['seat']==s]) for s in (0,1)], games=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(collect(args.revision), indent=2, allow_nan=False) + '\n')
