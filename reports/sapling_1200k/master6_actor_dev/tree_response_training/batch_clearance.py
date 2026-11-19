"""Certify preregistered continuation maps, with the original protected pair bound by identity."""
import argparse
import ast
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from clearance_build import ACTOR, BASE, HERE, ROOT, exam, load, make, mi, np, save, sha


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha(path.read_bytes())


def relative(path):
    return str(path.relative_to(ROOT))


def official_board(seed):
    np.random.seed(seed)
    game = make('carbon', configuration={**exam.CFG, 'randomSeed': seed})
    actual = mi.terrain_sha256(game.states[0].observation.full_carbon)
    check(actual == mi.local_board(seed), 'official/native opening disagreement')
    return actual


def reservations(protocol, plan):
    queue_path = HERE / 'next_intake.json'
    queue = load(queue_path)
    source = HERE / 'next_queue.py'
    script = source.read_text()
    assignments = [node for node in ast.parse(script).body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and getattr(node.targets[0], 'id', '') == 'NEXT_MAP_PAIRS']
    check(len(assignments) == 1, 'ambiguous schedule program')
    assignment = assignments[0]
    schedule = ast.literal_eval(assignment.value)
    before_after = script.splitlines()[:assignment.lineno - 1] + script.splitlines()[assignment.end_lineno:]
    check(not any(str(seed) in line for seed in plan['train_maps'] for line in before_after),
          'planned seed used outside literal schedule in program')
    check(digest(queue_path) == plan['queue_sha256'] and queue['state'] == 'registered_for_later_not_started'
          and queue['source_data_sha256'] == digest(HERE / 'training_labels.json')
          and queue['policy_sha256'] == plan['parent_pack_sha256'], 'wrong queue or policy')
    batches = queue['batches']
    check([list(pair) for pair in schedule] == [b['maps'] for b in batches], 'queue/program mismatch')
    index = next(i for i, b in enumerate(batches) if b['name'] == plan['intake_name'])
    entry = batches[index]
    check(protocol.parent.name == entry['name'] and plan['train_maps'] == entry['maps']
          and plan['training_job_registry'] == entry['jobs'] and entry['pending_full_historical_clearance'] is True
          and plan['development_maps'] == queue['protected_unopened_development_maps'], 'wrong batch registration')
    resource = protocol.parent / 'response_resources.json'
    resource_parent = HERE / 'response_resources.json'
    expected_resource = {**load(resource_parent), 'parent_protocol_sha256': digest(protocol),
                         'data_parent': digest(resource_parent), 'source_maps': entry['maps']}
    check(load(resource) == expected_resource, 'resource note contains undeclared uses')
    paths = [source, queue_path, protocol, resource]
    # Only genuine pre-declarations may explain nominal hits. An observed tape never qualifies.
    for path in paths:
        obj = subprocess.run(['git', 'show', plan['source_revision'] + ':' + relative(path)],
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        check(obj.returncode or sha(obj.stdout) == digest(path), 'frozen reservation changed: ' + str(path))
    return entry, [HERE] + [HERE / 'batches' / b['name'] for b in batches[:index]], {
        relative(path): digest(path) for path in paths}


def audit_previous(dirs, plan):
    past = {}
    for directory in dirs:
        reg, clear, receipt = [directory / name for name in ('protocol.json', 'clearance.json', 'source_receipt.json')]
        official = directory / 'source' / 'official_replay.json'
        reg_data, c, data = load(reg), load(clear), load(receipt)
        check(plan['prior_receipts'][str(directory.relative_to(HERE))] == digest(directory / 'training_labels.json'),
              'changed prior intake labels')
        check(c['verdict'] == 'eligible_in_checked_scope' and c['declaration_sha256'] == digest(reg)
              and data['clearance_sha256'] == digest(clear) and data['protocol_sha256'] == digest(reg)
              and data['official_sha256'] == digest(official), 'unbound prior source signatures')
        sources = data['games']
        expected = {(s, p) for s in reg_data['train_maps'] for p in (0, 1)}
        check(len(sources) == len(expected) and {(g['seed'], g['seat']) for g in sources} == expected,
              'incomplete prior source games')
        official_rows = load(official)['runs']['source']['games']
        by_pair = {(g['seed'], g['seat']):g for g in official_rows}
        check(len(official_rows) == len(expected) and set(by_pair) == expected, 'wrong official prior games')
        attest = []
        for row in sources:
            key = row['seed'], row['seat']
            replay = by_pair[key]
            tape = ROOT / row['raw_path']
            actual = load(tape)
            terrain = mi.terrain_sha256(actual['initial_full_carbon'])
            check(row['raw_sha256'] == replay['transcript_sha256'] == digest(tape)
                  and terrain == row['map_sha256'] == replay['map_sha256'] == c['boards'][str(key[0])]
                  and key == (actual['seed'], actual['candidate_seat']), 'prior game identity mismatch')
            attest.append({'seed':key[0], 'seat':key[1], 'terrain':terrain, 'tape':relative(tape),
                           'tape_sha256':row['raw_sha256']})
        past[relative(directory)] = {'source_receipt_sha256':digest(receipt),'official_sha256':digest(official),
                                     'protocol_sha256':digest(reg), 'clearance_sha256':digest(clear), 'games':attest}
    return past


def ladder():
    fingerprint, seeds, terrains = mi.ladder_boards()
    seeds, terrains = set(seeds), set(terrains)
    registered = set()
    files = {}
    for name, *_ in mi.SOURCES:
        path = BASE / name
        files[relative(path)] = digest(path)
        registered.update(json.loads(line)['id'] for line in gzip.decompress(path.read_bytes()).splitlines())
    extra = {}
    for replay in sorted((ROOT.parent / 'sapling-all-top5-20261119/replays').glob('*.json.gz')):
        gid = replay.name.split('.')[0]
        if gid not in registered:
            data = load(replay)
            opening = data['steps'][0][0]['observation']
            check(opening['step'] == 0, 'cached replay lacks opening')
            seed = data['configuration']['randomSeed']
            terrain = mi.terrain_sha256(opening['full_carbon'])
            seeds.add(seed)
            terrains.add(terrain)
            extra[gid] = {'seed':seed, 'terrain':terrain, 'file_sha256':digest(replay)}
    return seeds, terrains, {'registered_games':len(registered), 'registered_sha256':fingerprint,
                            'ledger_files_sha256':files, 'extra':extra}


def historical_catalogs(initial, boards):
    """Include removed-workspace evidence retained by the initial cross-project audit."""
    terrains, seeds = set(), set()
    for relative_name, record in initial['prior_catalogs'].items():
        path = ROOT / relative_name
        check(digest(path) == record['sha256'], 'historical catalog changed: ' + relative_name)
        old = load(path)
        for row in old.get('local_native_or_training_maps', []):
            terrains.add(row['terrain'])
            if row['seed'] is not None:
                seeds.add(row['seed'])
        named = old.get('boards', {})
        if isinstance(named, dict):
            for key, value in named.items():
                if isinstance(value, dict):
                    value = value.get('terrain_sha256', value.get('initial_terrain_u8_sha256'))
                if str(key).isdigit() and isinstance(value, str) and len(value) == 64:
                    terrains.add(value)
                    seeds.add(int(key))
    check(len(terrains) == initial['historical_terrains'] and len(seeds) == initial['historical_nominal_seeds'],
          'historical catalog cardinality changed')
    check(not (set(boards.values()) & terrains or set(map(int, boards)) & seeds),
          'map overlaps retained historical catalogs')
    return {'distinct_terrains':len(terrains), 'nominal_seeds':len(seeds),
            'source_sha256':{name:r['sha256'] for name,r in initial['prior_catalogs'].items()}}


def validate_scan(raw, seed, terrain, protocol, plan, declarations):
    row = json.loads(raw)
    allowed = set(declarations)
    local_allowed = {str((ROOT / path).relative_to(ROOT.parent)) for path in allowed}
    check(row['candidate_seed'] == seed and row['frozen_git_revision'] == plan['source_revision']
          and row['current_declaration']['sha256'] == digest(protocol)
          and row['candidate_initial_terrain_u8_sha256'] == row['official_opening_sha256'] == terrain
          and not row['reserved'] and not row['ladder_seed_or_terrain_match'], 'wrong scan or reserved map')
    for conflict in row['conflicts']:
        path, how, *extra = conflict
        if path == 'frozen_git' and how == 'unexpected_exact_references' and set(extra[0]) <= allowed:
            continue
        check(path in local_allowed and how in {'exact_number_or_terrain','recorded_seed'} and not extra,
              f'historical conflict: {conflict}')
    for board, nominal, method in row['same_terrain_matches']:
        check((board, nominal, method) == (terrain, seed, 'inferred_default_native_generator'),
              f'previously observed board: {board}, {nominal}, {method}')
        records = [x for x in row['local_native_or_training_maps']
                   if (x['terrain'], x['seed'], x['method']) == (board, nominal, method)]
        check(len(records) == 1 and records[0]['source'] and set(records[0]['source']) <= local_allowed,
              'same terrain has evidence beyond the declared reservation')
        for loc in records[0]['source']:
            check(row['local_source_sha256'][loc] == declarations[relative(ROOT.parent / loc)],
                  'queued terrain evidence identity changed')
    check(row['verdict'] == 'eligible_in_checked_scope' or row['conflicts'] or row['same_terrain_matches'],
          'unexplained generic denial')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol',type=Path)
    parser.add_argument('--work',type=Path,help='reuse full scans matching the declared map and Git revision')
    args = parser.parse_args()
    protocol = args.protocol.resolve()
    d = protocol.parent
    check(protocol == HERE / 'batches' / d.name / 'protocol.json', 'expected a registered batch protocol')
    plan = load(protocol)
    entry, prior_dirs, declarations = reservations(protocol, plan)
    previous = audit_previous(prior_dirs, plan)
    train, protected = plan['train_maps'], plan['development_maps']
    check(len(train) == 2 and len(protected) == 2 and len(set(train + protected)) == 4, 'duplicate seed')
    boards = {str(s):official_board(s) for s in train + protected}
    check(len(set(boards.values())) == len(boards), 'two registered seeds generate the same terrain')
    first = load(HERE / 'clearance.json')
    legacy = historical_catalogs(first, boards)
    check(protected == load(HERE / 'protocol.json')['development_maps']
          and all(boards[str(s)] == first['boards'][str(s)] for s in protected)
          and all(boards[str(s)] == entry['terrain_at_queue_time'][str(s)] for s in train),
          'protected or scheduled map has changed')
    prior_games = [row for source in previous.values() for row in source['games']]
    check(not ({row['seed'] for row in prior_games} & set(train + protected))
          and not ({row['terrain'] for row in prior_games} & set(boards.values())), 'prior real source map reuse')
    seeds, terrains, ladder_report = ladder()
    check(not (seeds & set(train+protected)) and not (terrains & set(boards.values())), 'public ladder overlap')
    final = d / 'clearance.json'
    if final.exists():
        receipt = load(final)
        check(receipt['declaration_sha256'] == digest(protocol) and receipt['boards'] == boards
              and receipt['prior_sources'] == previous and receipt['reservations_sha256'] == declarations
              and receipt['verdict'] == 'eligible_in_checked_scope', 'existing batch receipt changed')
        for item in receipt['full_audits'].values():
            check(digest(d / item['archive']) == item['sha256'], 'existing batch archive changed')
        return
    check(not any((d / 'source').glob('raw_*')), 'intake already played before clearance')
    scanner = ACTOR / 'freight_deployment/map_preflight.py'
    catalog = ACTOR / 'freight_deployment/first_response/map_preflight.json'
    with tempfile.TemporaryDirectory(prefix='sapling-batch-clearance-') as tmp:
        work = args.work.resolve() if args.work else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        audits, archives = {}, {}
        for s in train:
            path = work / f'{s}.json'
            if not path.exists():
                subprocess.run([sys.executable,str(scanner),'--seed',str(s),'--revision',plan['source_revision'],
                                '--declaration',relative(protocol),'--registry',str(catalog),'--out',str(path)],check=True,
                               env={**os.environ,'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'grep.threads','GIT_CONFIG_VALUE_0':'1'})
            raw = path.read_bytes()
            row = validate_scan(raw,s,boards[str(s)],protocol,plan,declarations)
            check(row['sha256_of_this_script'] == digest(scanner) and
                  row['historical_catalog_loaded_sha256'] == digest(catalog), 'wrong scanner/catalog')
            archive = d / f'clearance_audit_{s}.zip'
            payload = Path(tmp) / archive.name
            with zipfile.ZipFile(payload,'w') as out:
                info = zipfile.ZipInfo(f'{s}.json')
                info.compress_type = zipfile.ZIP_DEFLATED
                out.writestr(info,raw)
            archives[archive] = payload
            audits[str(s)] = {'archive': archive.name,'sha256':digest(payload),'report_sha256':sha(raw),
                              'coverage':row['coverage'],'frozen_ladder':row['frozen_ladder'],
                              'generic_nominal_conflicts':row['conflicts'],
                              'generic_terrain_matches':row['same_terrain_matches']}
        for target, staged in archives.items():
            dest = target.with_suffix('.tmp')
            dest.write_bytes(staged.read_bytes())
            dest.replace(target)
        save(final,{'stage':'dynamic_observed_maps_and_historical_seed_ranges','verdict':'eligible_in_checked_scope',
                    'source_revision':plan['source_revision'],'declaration_sha256':digest(protocol),'boards':boards,
                    'historical_catalogs':legacy,
                    'training_maps':train,'protected_identity_only':protected,
                    'initial_clearance_sha256':digest(HERE/'clearance.json'),'prior_sources':previous,
                    'reservations_sha256':declarations,'ladder':ladder_report,'full_audits':audits,
                    'scope':'Full seed/range/terrain scans of training maps. Reserved development openings are regenerated only for identity and collision checks, without playing. Nominal schedule references are allowed only with bound program, queue, declaration or resource bytes; observed boards cannot qualify.',
                    'audit_code_sha256':digest(Path(__file__))})
        print('eligible_in_checked_scope',digest(final),boards,flush=True)


if __name__ == '__main__':
    main()
