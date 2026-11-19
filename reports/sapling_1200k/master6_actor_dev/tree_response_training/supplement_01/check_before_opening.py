"""Recheck an unopened training batch and its held-out boards against a later published ladder cutoff."""
import argparse
import gzip
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import batch_clearance as audit
from select_roots import load,retain,sha


def verify(area,minimum_ids):
    clearance=load(area/'clearance.json');plan=load(area/'protocol.json')
    assert_before=not (area/'source/results.json').exists() and not any((area/'source').glob('raw_*'))
    if not assert_before or clearance['declaration_sha256']!=sha(area/'protocol.json'):
        raise ValueError('requires the registered, not-yet-played training batch')
    excluded_seeds,excluded_terrains,public=audit.ladder()
    files={};ids=set();seeds=set();terrains=set()
    for name,*_ in audit.mi.SOURCES:
        path=audit.BASE/name
        rows=[json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
        if any(r['id'] in ids for r in rows):raise ValueError('duplicate observed ladder game')
        ids.update(r['id'] for r in rows);seeds.update(r['seed'] for r in rows)
        terrains.update(r['initial_terrain_u8_sha256'] for r in rows);files[name]=sha(path)
    if len(ids)<minimum_ids or not seeds<=excluded_seeds or not terrains<=excluded_terrains:raise ValueError('incomplete actual map census')
    plan_ids=plan['train_maps']+plan['development_maps']
    boards={str(s):audit.official_board(s) for s in plan_ids}
    if (boards!=clearance['boards'] or len(boards)!=4 or len(set(boards.values()))!=4 or set(plan_ids)&excluded_seeds
            or set(boards.values())&excluded_terrains):raise ValueError('new ladder overlaps or substitutes a registered board')
    # Record the declared configuration by hash; avoid propagating further literal seed references.
    return {'schema':'supplement-map-exposure-recheck-before-playing-v1','protocol_sha256':sha(area/'protocol.json'),
            'earlier_clearance_sha256':sha(area/'clearance.json'),'prior_registered_ladder_count':clearance['ladder']['registered_games'],
            'ladder_ids_verified':len(ids),'registered_seed_count':len(seeds),'registered_terrain_count':len(terrains),
            'additionally_cached_replays':len(public['extra']),'registered_ladder_digest':public['registered_sha256'],
            'ladder_source_file_hashes':files,'training_and_heldout_board_count':len(boards),'source_games_issued':0,
            'official_vs_registered_boards_match':True,'intersection_by_seed':0,'intersection_by_observed_terrain':0,
            'checker_sha256':sha(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',choices=['intake_06','intake_07','intake_08']);parser.add_argument('--minimum-ids',type=int,required=True)
    a=parser.parse_args();area=HERE/'batches'/a.batch
    verified=verify(area,a.minimum_ids);path=area/f"opening_recheck_{verified['ladder_ids_verified']}.json"
    retain(path,verified);print(path,sha(path))

if __name__=='__main__':main()
