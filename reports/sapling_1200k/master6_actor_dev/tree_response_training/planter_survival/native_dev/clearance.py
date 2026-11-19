"""Verify independent native-development openings against registered, played and reserved terrain."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
sys.path.insert(0, str(TREE))
import batch_clearance as prior


def exclusion_ledger():
    paths = [TREE / 'next_intake.json', TREE / 'supplement_01/registration.json']
    named = dict(prior.load(TREE / 'clearance.json')['boards'])
    files = {}
    for path in paths:
        if not path.exists():
            continue
        files[prior.relative(path)] = prior.digest(path)
        for batch in prior.load(path)['batches']:
            for name, terrain in batch['terrain_at_queue_time'].items():
                prior.check(name not in named, 'overlap in two independent prior intakes')
                named[name] = terrain
    transcripts = 0
    for path in sorted(TREE.glob('**/source_receipt.json')):
        if path.parent == HERE:
            continue
        source = prior.load(path)
        if 'games' not in source or 'official_sha256' not in source:
            continue
        official_file = path.parent / 'source/official_replay.json'
        prior.check(prior.digest(official_file) == source['official_sha256'], 'official source receipt changed')
        files[prior.relative(path)] = prior.digest(path)
        official = prior.load(official_file)['runs']['source']['games']
        proofs = {(x['seed'], x['seat']):x for x in official}
        for game in source['games']:
            replay = proofs[game['seed'], game['seat']]
            tape_file = prior.ROOT / game['raw_path']
            tape = prior.load(tape_file)
            true_terrain = prior.mi.terrain_sha256(tape['initial_full_carbon'])
            prior.check(true_terrain == game['map_sha256'] == replay['map_sha256'] and
                        prior.digest(tape_file) == game['raw_sha256'] == replay['transcript_sha256'],
                        'prior game has wrong initial board or unverified tape')
            key = str(game['seed'])
            prior.check(named.get(key, true_terrain) == true_terrain, 'previous board differs from registration')
            named[key] = true_terrain
            transcripts += 1
    return named, files, transcripts


def issue(input_dir):
    protocol = HERE / 'protocol.json'
    plan = prior.load(protocol)
    maps = plan['maps']
    prior.check(len(maps) == len(set(maps)) == 2 and plan['physical_seats'] == [0,1], 'incorrect native development scope')
    final = HERE / 'clearance.json'
    if final.exists():
        previous = prior.load(final)
        prior.check(previous['declaration_sha256'] == prior.digest(protocol) and
                    previous['verdict'] == 'eligible_in_checked_scope' and
                    all(prior.digest(HERE / rec['archive']) == rec['sha256'] for rec in previous['full_audits'].values()),
                    'sealed clearance does not match the active protocol')
        return
    boards = {str(seed):prior.official_board(seed) for seed in maps}
    cached_seeds, cached_boards, ladder = prior.ladder()
    historical = prior.historical_catalogs(prior.load(TREE / 'clearance.json'), boards)
    existing, files, source_games = exclusion_ledger()
    prior.check(len(set(boards.values())) == len(boards)
                and not (set(maps) & cached_seeds or set(boards.values()) & cached_boards)
                and not (set(boards) & set(existing) or set(boards.values()) & set(existing.values())),
                'terrain or seed overlaps a previous map')
    scanner = prior.ACTOR / 'freight_deployment/map_preflight.py'
    registry = scanner.parent / 'first_response/map_preflight.json'
    audits = {}
    staged = []
    with tempfile.TemporaryDirectory(prefix='survival-native-audit-') as tmp:
        for seed in maps:
            output = (input_dir or Path(tmp)) / f'{seed}.json'
            if not output.exists():
                subprocess.run([sys.executable, str(scanner), '--seed',str(seed),'--revision',plan['source_revision'],
                                '--declaration',prior.relative(protocol),'--registry',str(registry),'--out',str(output)],
                               check=True, env={**os.environ,'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'grep.threads',
                                                'GIT_CONFIG_VALUE_0':'1'})
            raw = output.read_bytes()
            r = prior.validate_scan(raw,seed,boards[str(seed)],protocol,plan,{prior.relative(protocol):prior.digest(protocol)})
            prior.check(not r['conflicts'] and not r['same_terrain_matches'] and r['verdict'] == 'eligible_in_checked_scope'
                        and r['historical_catalog_loaded_sha256'] == prior.digest(registry)
                        and r['sha256_of_this_script'] == prior.digest(scanner),'unexpected historical conflict or scanner')
            archive = Path(tmp)/f'clearance_audit_{seed}.zip'
            member = zipfile.ZipInfo(output.name)
            member.compress_type = zipfile.ZIP_DEFLATED
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr(member,raw)
            audits[str(seed)] = {'archive':archive.name,'sha256':prior.digest(archive),'report_sha256':prior.sha(raw),
                                 'coverage':r['coverage'],'frozen_ladder':r['frozen_ladder'],
                                 'reserved_formal_half_open':r['formal_reserved_half_open']}
            staged.append(archive)
        for archive in staged:
            dst = HERE / archive.name
            draft = dst.with_suffix('.tmp')
            draft.write_bytes(archive.read_bytes())
            draft.replace(dst)
    prior.save(final,{'verdict':'eligible_in_checked_scope','declaration_sha256':prior.digest(protocol),
                      'source_revision':plan['source_revision'],'boards':boards,'full_audits':audits,'ladder':ladder,
                      'original_catalogs':historical,'source_receipts_and_queues_sha256':files,
                      'actual_transcripts_cross_checked':source_games,'prior_deduplicated_reserved_or_played_maps':len(existing),
                      'scope':'Two independent development maps for the survival veto. Both historical and prospective training, previous held-out maps, public observed terrain, listed workspaces and the formal reservation are excluded.','audit_issuer_sha256':prior.digest(Path(__file__))})
    print('eligible_in_checked_scope',prior.digest(final),boards,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,help='directory containing the original scans for this frozen declaration')
    issue(parser.parse_args().work)
