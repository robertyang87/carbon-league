"""Rebuild paid first-tenure samples from the archived verified cash-event ledgers."""
from pathlib import Path
import gzip
import hashlib
import json
import tarfile
import numpy as np

HERE = Path(__file__).resolve().parent
TOP = HERE.parents[1] / 'all_top5'
LAST = TOP / 'appendix_4a4677c9'
FEATURES = ['kind', 'price', 'neighbor_carbon', 'cash_after_price', 'remaining_turns', 'old_tree_age',
            'opp_planter_dist', 'own_collector_dist', 'opp_collector_dist', 'alt_adjacent_enemy_tree',
            'own_trees', 'alive_global_trees', 'own_home_dist', 'opp_home_dist']
SPLITS = ['train', 'calibration', 'held_map', 'held_opponent']
REASONS = ['expired', 'seized', 'terminal']


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(p):
    with (gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()) as f:
        return json.load(f)


def identities():
    tail = read(LAST / 'manifest.json')
    chain = tail['baseline_revisions'] + [tail['revision']]
    manifests = {}
    for p in [TOP / 'manifest.json', *TOP.glob('appendix_*/manifest.json')]:
        m = read(p)
        if m['revision'] in chain:
            manifests[m['revision']] = (p, m)
    return [manifests[r] for r in chain]


def source_records(path):
    if path.parent == TOP:
        for p in sorted((TOP / 'plant_timing_data').glob('*.tar')):
            with tarfile.open(p) as archive:
                for member in archive.getmembers():
                    row = json.loads(gzip.decompress(archive.extractfile(member).read()))
                    yield row['id'], row['replay_sha256'], row
    else:
        for row in read(path.parent / 'cases.json.gz')['rows']:
            yield row['id'], row['source_sha256'], row['planting']


def build(cache):
    manifests = identities()
    schema = read(TOP / 'plant_timing_schema.json')
    ix = [schema['choices'].index(k) for k in FEATURES]
    elite = {p['name'] for p in manifests[0][1]['top5']}
    maps, x, y, ixmap, seats, reasons, roots, cells = [], [], [], [], [], [], [], []
    seen, known_terrain, known_seed = set(), set(), set()
    names = {n for _, m in manifests for g in m['games'] for n in g['seat_names']}
    held_names = {n for n in names - elite if int(h('tenure-opponent-v1|' + n), 16) % 5 == 0}
    source_receipts = []
    for path, m in manifests:
        cards = {r['id']: r for r in m['games'] if r['frame_verified'] and all(isinstance(v,(int,float)) for v in r['score_by_seat'])}
        terrain = {r['id']: r['initial_terrain_u8_sha256'] for r in map(json.loads, gzip.open(path.parent / 'validation_maps.jsonl.gz', 'rt'))}
        linked = set()
        for gid, sha, ledger in source_records(path):
            g = cards[gid]
            assert gid not in linked and gid not in seen and g['sha256'] == sha and ledger['rewards'] == g['score_by_seat']
            assert ledger['turns'] == g['transitions'] and ledger['seed'] == g['seed'] and g['seed'] not in known_seed
            layout = terrain[gid]
            assert layout not in known_terrain
            known_seed.add(g['seed']); known_terrain.add(layout); linked.add(gid); seen.add(gid)
            bucket = int(h('tenure-map-v1|' + layout), 16) % 8
            group = 3 if held_names.intersection(g['seat_names']) else {0:2, 1:1}.get(bucket, 0)
            mi = len(maps)
            maps.append(dict(id=gid, sha256=sha, terrain=layout, seed=g['seed'], names=g['seat_names'], split=SPLITS[group],
                             held_rival=[g['seat_names'][1-s] in held_names for s in (0,1)], turns=g['transitions']))
            purchases = ledger['purchases']; indices = set()
            for c in ledger['choices']:
                j = c[15]
                if j is None:
                    continue
                p = purchases[j]
                assert j not in indices and (c[0],c[1],c[2],c[3],c[4]) == (p[0],p[1],p[2],p[4],p[5])
                assert abs(c[5]-p[6]) < .005 and 0 <= p[10] <= c[8] and p[12] <= p[11] + .0001
                indices.add(j)
                row = [c[i] for i in ix]; row[0] = int(row[0] == 'seize')
                x.append(row); y.append([p[10], p[11]+p[13], p[12], p[13], p[6]])
                ixmap.append(mi); seats.append(c[1]); reasons.append(REASONS.index(p[9])); roots.append(c[0]); cells.append(c[3])
            assert len(indices) == len(purchases)
        assert linked == set(cards), (path, len(linked), len(cards))
        source_receipts.append(dict(revision=m['revision'], verified_games=len(linked), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert not held_names.intersection(n for r in maps if r['split'] != 'held_opponent' for n in r['names'])
    np.savez_compressed(cache, x=np.asarray(x, dtype=np.float32), y=np.asarray(y,dtype=np.float64),
                        map=np.asarray(ixmap,dtype=np.int32), seat=np.asarray(seats,dtype=np.int8), reason=np.asarray(reasons,dtype=np.int8),
                        root=np.asarray(roots,dtype=np.int16), cell=np.asarray(cells,dtype=np.int16))
    out = dict(sources=source_receipts, maps=maps, reserved_names=sorted(held_names), features=FEATURES, splits=SPLITS, reasons=REASONS)
    (HERE/'maps.json.gz').write_bytes(gzip.compress(json.dumps(out, separators=(',',':')).encode(), mtime=0))
    print('maps and paid spells', len(maps), len(x), flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('cache', type=Path)
    build(parser.parse_args().cache)
