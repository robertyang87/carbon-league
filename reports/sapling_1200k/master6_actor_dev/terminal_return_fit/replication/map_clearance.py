"""Combine both pre-game receipts; compare older and newly staged public boards separately."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FIT = HERE.parent
ACTOR = FIT.parent
BASE = ACTOR.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
import map_identity


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def summarize(src):
    protocol=load(HERE/'protocol.json')
    seeds=protocol['maps']
    prior=load(FIT/'map_clearance.json')['boards']
    if set(map(int,prior)) != {protocol['prior_training_seed'],protocol['prior_development_trigger_seed']}:
        raise ValueError('the prior stage and registered replication disagree')
    old_reports = sorted(FIT.glob('map_audit_*.json.gz'))
    if len(old_reports)!=2:
        raise ValueError('expected the two recorded training/trigger map audits')
    prior_terrain_records={r['terrain'] for p in old_reports for r in load(p)['local_native_or_training_maps']}
    prior_nominal={r['seed'] for p in old_reports for r in load(p)['local_native_or_training_maps'] if r['seed'] is not None}
    ladder_sha, ladder_seeds, ladder_boards=map_identity.ladder_boards()
    # Registry extensions in flight may already have actual replays in the cache.
    pending=[]
    extra={}
    extra_inputs={}
    intake_counts={}
    for path in sorted((BASE/'all_top5').glob('intake_*.json')):
        payload=load(path)
        extra_inputs[path.name]=digest(path.read_bytes())
        intake_counts[path.name]={'cached_already_in_registry':0,'cached_outside_registry':0,'unavailable':0}
        for card in payload.get('games',[]):
            if not isinstance(card,dict) or 'id' not in card:
                continue
            gid=card['id']
            replay=ROOT.parent/'sapling-all-top5-20261119/replays'/f'{gid}.json.gz'
            if not replay.exists():
                pending.append(gid)
                intake_counts[path.name]['unavailable']+=1
                continue
            raw=replay.read_bytes()
            game=json.loads(gzip.decompress(raw))
            ob=game['steps'][0][0]['observation']
            if ob['step']!=0:
                raise ValueError('missing public opening')
            seed=game['configuration']['randomSeed']
            terrain=map_identity.terrain_sha256(ob['full_carbon'])
            if seed in ladder_seeds and terrain in ladder_boards:
                intake_counts[path.name]['cached_already_in_registry']+=1
                continue
            intake_counts[path.name]['cached_outside_registry']+=1
            extra[gid]={'seed':seed,'terrain':terrain,'raw_sha256':digest(raw)}
    details={}
    boards={}
    raw_outputs={}
    for seed in seeds:
        path=src/f'{seed}.json'
        raw=path.read_bytes()
        audit=json.loads(raw)
        terrain=audit['candidate_initial_terrain_u8_sha256']
        declaration=audit['current_declaration']
        if (audit['candidate_seed']!=seed or audit['frozen_git_revision']!=protocol['source_revision'] or
            declaration['sha256']!=digest((HERE/'protocol.json').read_bytes()) or
            audit['verdict']!='eligible_in_checked_scope' or audit['conflicts'] or audit['same_terrain_matches'] or
            audit['reserved'] or audit['ladder_seed_or_terrain_match'] or
            terrain!=audit['official_opening_sha256'] or terrain!=map_identity.local_board(seed) or
            terrain in set(prior.values())|set(boards.values())|prior_terrain_records|ladder_boards or
            seed in set(map(int,prior))|prior_nominal|ladder_seeds or
            any(v['seed']==seed or v['terrain']==terrain for v in extra.values())):
            raise ValueError(f'failed seed, board identity, or exposure check for {seed}: {audit["conflicts"]}')
        boards[str(seed)]=terrain
        filename=f'map_clearance_{seed}.json.gz'
        raw_outputs[filename]=gzip.compress(raw,mtime=0)
        details[str(seed)]={'full_precompetition_audit':filename,'uncompressed_sha256':digest(raw),
                            'compressed_sha256':digest(raw_outputs[filename]),
                            'verified_ledger_games_at_scan':audit['ladder']['games'],
                            'distinct_nonladder_recorded_terrains_at_scan':audit['coverage']['distinct_nonladder_recorded_terrains'],
                            'distinct_nonladder_inferred_terrains_at_scan':audit['coverage']['distinct_nonladder_inferred_terrains'],
                            'unique_searched_file_content_at_scan':audit['coverage']['unique_file_or_member_payloads']}
    receipt={'verdict':'eligible_in_checked_scope','boards':boards,'source_revision':protocol['source_revision'],
             'declaration_sha256':digest((HERE/'protocol.json').read_bytes()),'prior_training_and_trigger':prior,
             'old_full_audit_sha256':{p.name:digest(p.read_bytes()) for p in old_reports},
             'old_reports_distinct_terrain_identities':len(prior_terrain_records),
             'old_reports_distinct_nominal_seeds':len(prior_nominal),'ladder_ledger_sha256':ladder_sha,
             'ladder_distinct_seed_count':len(ladder_seeds),'ladder_distinct_board_count':len(ladder_boards),
             'available_additional_replays':extra,'additional_intake_source_hash':extra_inputs,'intake_coverage':intake_counts,
             'cards_without_usable_opening':sorted(set(pending)-set(extra)),
             'details':details,'limits':'Missing score-card openings and unpublished remote maps have unknown identity. Old generator-inferred maps are distinguished from recorded boards inside each full audit.'}
    for file,data in raw_outputs.items():
        (HERE/file).write_bytes(data)
    output=HERE/'map_clearance.json'
    tmp=output.with_suffix('.tmp')
    tmp.write_text(json.dumps(receipt,indent=2)+'\n')
    tmp.replace(output)
    print(json.dumps({k:receipt[k] for k in ('verdict','boards','ladder_distinct_board_count','cards_without_usable_opening')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('preflight_directory',type=Path,help='directory containing one full generic preflight result <seed>.json per registered replication map')
    summarize(ap.parse_args().preflight_directory)
