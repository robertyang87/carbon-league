"""Check the separately registered training supplement after the previous intake is sealed."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
TREE = HERE.parent
sys.path.insert(0, str(TREE))
import batch_clearance as prior


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_schedule(program):
    text = program.read_text()
    nodes = [node for node in ast.parse(text).body if isinstance(node, ast.Assign) and
             len(node.targets) == 1 and getattr(node.targets[0], 'id', None) == 'NEXT_MAP_PAIRS']
    require(len(nodes) == 1, 'expected a single NEXT_MAP_PAIRS assignment')
    node = nodes[0]
    pairs = ast.literal_eval(node.value)
    require(len(pairs) == 3 and all(len(p) == 2 for p in pairs), 'invalid supplement pair registry')
    seeds = [s for pair in pairs for s in pair]
    require(len(set(seeds)) == len(seeds) and all(type(s) is int for s in seeds),
            'supplement has duplicate or nonintegral map identifiers')
    other_lines = text.splitlines()[:node.lineno - 1] + text.splitlines()[node.end_lineno:]
    require(not any(str(seed) in line for line in other_lines for seed in seeds),
            'an allocated identifier is used outside the pure schedule literal')
    return [list(pair) for pair in pairs]


def declared_paths(paths, revision):
    result = {}
    for path in paths:
        name = prior.relative(path)
        checksum = prior.digest(path)
        stored = subprocess.run(['git', 'show', revision + ':' + name], cwd=prior.ROOT,
                                capture_output=True)
        if stored.returncode == 0:
            require(hashlib.sha256(stored.stdout).hexdigest() == checksum,
                    'frozen declarations changed: ' + name)
        result[name] = checksum
    return result


def previous_chain(dirs, plan):
    labels = {str(path.relative_to(TREE)): prior.digest(path / 'training_labels.json') for path in dirs}
    require(plan['prior_receipts'] == labels, 'prior map evidence chain mismatch')
    # Every continuation prefix is sealed at its own end, including the original fifth intake.
    for idx, path in enumerate(dirs[1:], 1):
        aggregate = prior.load(path / 'aggregate.json')
        expected = [{'dir': name, 'labels_sha256': digest} for name, digest in list(labels.items())[:idx + 1]]
        require(aggregate['source_intakes'] == expected and aggregate['policy_sha256'] == plan['parent_pack_sha256']
                and aggregate['held_out_games'] == 0, 'invalid prior sealed aggregate: ' + str(path))
    return prior.load(dirs[-1] / 'aggregate.json')


def validate_registration(protocol, plan):
    allocator = HERE / 'register.py'
    registration = HERE / 'registration.json'
    reg = prior.load(registration)
    original_queue = TREE / 'next_intake.json'
    baseline = prior.load(TREE / 'protocol.json')
    original_dirs = [TREE] + [TREE / 'batches' / b['name'] for b in prior.load(original_queue)['batches']]
    require([p.name for p in original_dirs[1:]] == ['intake_02', 'intake_03', 'intake_04', 'intake_05'],
            'wrong original intake sequence')
    require(reg['original_05_must_seal'] is True and
            reg['expected_original_05_prefix'] == [str(p.relative_to(TREE)) for p in original_dirs],
            'supplement skips the original fifth intake')
    require(reg['schema'] == 'sapling-tree-training-supplement-v1' and
            reg['register_program_sha256'] == prior.digest(allocator) and
            reg['original_protocol_sha256'] == prior.digest(TREE / 'protocol.json') and
            reg['original_queue_sha256'] == prior.digest(original_queue) and
            reg['original_03_aggregate_sha256'] == prior.digest(original_dirs[2] / 'aggregate.json') and
            plan['queue_sha256'] == prior.digest(registration), 'wrong original or supplementary registration')
    require(plan['parent_pack_sha256'] == baseline['parent_pack_sha256'] == reg['policy_sha256']
            and plan['development_maps'] == baseline['development_maps'] == reg['protected_unopened_development_maps'],
            'policy or protected maps were replaced')
    batches = reg['batches']
    require([b['name'] for b in batches] == ['intake_06', 'intake_07', 'intake_08'] and
            [b['maps'] for b in batches] == source_schedule(allocator), 'wrong supplementary batch registry')
    idx = [b['name'] for b in batches].index(plan['intake_name'])
    batch = batches[idx]
    require(protocol.parent.name == batch['name'] and plan['train_maps'] == batch['maps'] and
            batch['pending_full_historical_clearance'] is True and plan['training_job_registry'] == batch['jobs'],
            'wrong job registry or training maps')
    past_dirs = original_dirs + [HERE / 'batches' / b['name'] for b in batches[:idx]]
    last = previous_chain(past_dirs, plan)
    require(last['distinct_maps'] < reg['goals']['independent_training_maps'] or
            last['filled_pairs'] < reg['goals']['completed_actually_paid_pairs'],
            'existing intake already satisfies both registered data counts')
    resource = protocol.parent / 'response_resources.json'
    base_resource = TREE / 'response_resources.json'
    expected_resource = {**prior.load(base_resource), 'parent_protocol_sha256': prior.digest(protocol),
                         'data_parent': prior.digest(base_resource), 'source_maps': batch['maps']}
    require(prior.load(resource) == expected_resource, 'unexpected supplementary resource declaration')
    revision = plan['source_revision']
    require(subprocess.run(['git', 'rev-parse', '--verify', revision + '^{commit}'], cwd=prior.ROOT,
                           capture_output=True).returncode == 0, 'unavailable freeze revision')
    return batch, past_dirs, declared_paths([allocator, registration, protocol, resource], revision)


def scan_training_map(seed, board, protocol, plan, allowed, temporary):
    scanner = prior.ACTOR / 'freight_deployment/map_preflight.py'
    catalog = prior.ACTOR / 'freight_deployment/first_response/map_preflight.json'
    output = temporary / f'{seed}.json'
    subprocess.run([sys.executable, str(scanner), '--seed', str(seed), '--revision', plan['source_revision'],
                    '--declaration', prior.relative(protocol), '--registry', str(catalog), '--out', str(output)],
                   check=True, env={**os.environ, 'GIT_CONFIG_COUNT':'1', 'GIT_CONFIG_KEY_0':'grep.threads',
                                    'GIT_CONFIG_VALUE_0':'1'})
    raw = output.read_bytes()
    report = prior.validate_scan(raw, seed, board, protocol, plan, allowed)
    require(report['historical_catalog_loaded_sha256'] == prior.digest(catalog) and
            report['sha256_of_this_script'] == prior.digest(scanner), 'full-audit inputs changed')
    archive = temporary / f'clearance_audit_{seed}.zip'
    member = zipfile.ZipInfo(output.name)
    member.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(archive, 'w') as target:
        target.writestr(member, raw)
    result = {'archive':archive.name, 'sha256':prior.digest(archive), 'report_sha256':prior.sha(raw),
              'coverage':report['coverage'], 'frozen_ladder':report['frozen_ladder'],
              'generic_nominal_conflicts':report['conflicts'], 'generic_terrain_matches':report['same_terrain_matches']}
    return result, archive


def certify(protocol):
    protocol = protocol.resolve()
    folder = protocol.parent
    require(protocol == HERE / 'batches' / folder.name / 'protocol.json', 'invalid supplement protocol path')
    plan = prior.load(protocol)
    batch, past_dirs, allowed = validate_registration(protocol, plan)
    previous = prior.audit_previous(past_dirs, plan)
    training, protected = plan['train_maps'], plan['development_maps']
    require(len(training) == len(protected) == 2 and len(set(training + protected)) == 4,
            'two different training maps and two protected maps are required')
    boards = {str(seed):prior.official_board(seed) for seed in training + protected}
    require(len(set(boards.values())) == 4, 'identical opening terrain under two identifiers')
    for seed in training:
        require(boards[str(seed)] == batch['terrain_at_queue_time'][str(seed)], 'registration board changed')
    first_clearance = TREE / 'clearance.json'
    first = prior.load(first_clearance)
    for seed in protected:
        require(boards[str(seed)] == first['boards'][str(seed)], 'protected map was replaced')
    local = prior.historical_catalogs(first, boards)
    games = [g for old in previous.values() for g in old['games']]
    names, terrains = set(training + protected), set(boards.values())
    require(not any(g['seed'] in names or g['terrain'] in terrains for g in games),
            'previously played training map or terrain matches a new or protected board')
    seen_seeds, seen_terrains, ladder = prior.ladder()
    require(not (names & seen_seeds or terrains & seen_terrains), 'new or protected map collides with public ladder')
    destination = folder / 'clearance.json'
    if destination.exists():
        saved = prior.load(destination)
        require(saved['declaration_sha256'] == prior.digest(protocol) and saved['boards'] == boards
                and saved['prior_sources'] == previous and saved['reservations_sha256'] == allowed
                and saved['registration_sha256'] == plan['queue_sha256']
                and saved['verdict'] == 'eligible_in_checked_scope'
                and set(saved['full_audits']) == set(map(str, training)), 'changed frozen supplementary receipt')
        for entry in saved['full_audits'].values():
            require(prior.digest(folder / entry['archive']) == entry['sha256'], 'changed frozen audit')
        return saved
    require(not any((folder / 'source').glob('raw_*')), 'source games predate map clearance')
    with tempfile.TemporaryDirectory(prefix='tree-supplement-clearance-') as temp:
        audits = {}
        archives = []
        for seed in training:
            audits[str(seed)], archive = scan_training_map(seed, boards[str(seed)], protocol, plan, allowed, Path(temp))
            archives.append(archive)
        for archive in archives:
            output = folder / archive.name
            draft = output.with_suffix('.tmp')
            draft.write_bytes(archive.read_bytes())
            draft.replace(output)
    receipt = {'stage':'supplement_observed_maps_and_historical_seed_ranges', 'verdict':'eligible_in_checked_scope',
               'source_revision':plan['source_revision'], 'declaration_sha256':prior.digest(protocol),
               'registration_sha256':plan['queue_sha256'], 'boards':boards, 'training_maps':training,
               'protected_identity_only':protected, 'initial_clearance_sha256':prior.digest(first_clearance),
               'prior_sources':previous, 'reservations_sha256':allowed, 'historical_catalogs':local, 'ladder':ladder,
               'full_audits':audits, 'audit_code_sha256':prior.digest(Path(__file__)),
               'shared_auditor_sha256':prior.digest(Path(prior.__file__)),
               'prior_aggregate_sha256':{str(area.relative_to(TREE)):prior.digest(area / 'aggregate.json')
                                         for area in past_dirs[1:]}}
    prior.save(destination, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol', type=Path)
    args = parser.parse_args()
    audit = certify(args.protocol)
    print(audit['verdict'], prior.digest(args.protocol.resolve().parent / 'clearance.json'), flush=True)
