"""Frozen cargo-defense vs parent development matches against original Master3s2."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
FAMILY = LANE.parents[1]
ROOT = next(p for p in HERE.parents if (p / '.git').exists())
STRATEGY = ROOT / 'reports/sapling_1200k/master6_actor_dev/strategic_destination_learning'
BASE = STRATEGY / 'development'
PREVIOUS = LANE.parent / 'prediction_feedback/development'
PLAN = HERE / 'protocol.json'
SEEDS = (621758594, 621758595)
ARMS = ('parent', 'defense')
WORKFLOW = 'master3s2-cargo-defense-development.yml'
EXPOSURE_STEP = 'Play parent and cargo-defense candidate against original'
SOURCE = '211e2e9aca9a0b3e50e2234ad203a4d93f344a502bddc4259142cf5de316aa37'
CANDIDATE = '2736b05f0e7ce3cf616eca8288efee87c491728d21c1fc0adae21721e81529e4'
PARENT = '7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed'
ORIGINAL = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def contract():
    reference = PREVIOUS / 'protocol.json'
    require(sha(reference) == SOURCE, 'Different reference tournament')
    previous = read(reference)
    keys = ('opponent', 'official_helpers', 'rules', 'environment', 'ladder_runner', 'worker', 'container',
            'base_registration', 'original_game_configuration', 'current_guide', 'match_runner', 'replayer',
            'native_timing_audit', 'previous_exposure_guard', 'registered_runner', 'registered_settler',
            'registered_sparse', 'observation_replay', 'parent', 'parent_weights')
    files = {key: previous['files'][key] for key in keys}
    require(all(runner.record(ROOT / card['path']) == card for card in files.values()), 'Original, parent or referee changed')
    require(files['opponent']['sha256'] == ORIGINAL and files['parent']['sha256'] == PARENT, 'Wrong baseline')
    pkg = LANE / 'package'
    paths = {'defense': pkg / 'candidate.py', 'defense_weights': LANE / 'weights.json',
             'manifest': pkg / 'manifest.json', 'builder': pkg / 'build.py', 'insert': pkg / 'insert.py',
             'runtime': LANE / 'runtime.py', 'fit_protocol': LANE / 'protocol.json', 'held': LANE / 'held.json',
             'parity': pkg / 'parity.json', 'probe': pkg / 'makabaka_roots.json.gz',
             'clearance': HERE / 'clearance.json', 'clearance_code': HERE / 'clearance.py',
             'matrix': HERE / 'matrix.py', 'workflow': ROOT / '.github/workflows' / WORKFLOW,
             'prior_reference': reference, 'paired_settler': PREVIOUS / 'settle.py'}
    files.update({key: runner.record(path) for key, path in paths.items()})
    manifest = read(paths['manifest'])
    held = read(paths['held'])
    parity = read(paths['parity'])
    probes = json.loads(gzip.decompress(paths['probe'].read_bytes()))
    require(files['defense']['sha256'] == manifest['candidate_sha256'] == CANDIDATE and
            manifest['parent_sha256'] == PARENT and files['defense']['bytes'] == manifest['bytes'] <= 10_000_000,
            'Cargo-defense package identity or size changed')
    for key, field in [('defense_weights', 'weight_sha256'), ('builder', 'build_sha256'),
                       ('runtime', 'runtime_sha256'), ('insert', 'insert_sha256')]:
        require(files[key]['sha256'] == manifest[field], 'Manifest disagrees for ' + key)
    require(held['verdict'] == 'PASS' and held['protocol'] == files['fit_protocol']['sha256'] and
            held['weight_sha256'] == manifest['weight_sha256'], 'Held cargo-risk predictor evidence mismatch')
    require(parity['candidate_sha256'] == probes['candidate_sha256'] == CANDIDATE and
            parity['source_parent_sha256'] == probes['parent_sha256'] == PARENT and
            parity['candidate_cells_without_new_friendly_conflicts'] and parity['checked_steps'] == 299 and
            probes['positive_qualifying'] == 2, 'Different physical action or parity receipt')
    census = read(paths['clearance'])
    boards = census['by_seed']
    require(census['passed'] and set(map(int, boards)) == set(SEEDS) and len(set(boards.values())) == len(SEEDS)
            and not census['physical_intersections'] and not census['seed_intersections']
            and census['boards'] == boards and census['verifier_sha256'] == files['clearance_code']['sha256'],
            'Uncleared map or different census implementation')
    schedule = [{'seed': seed, 'seat': seat, 'order': list(ARMS if (i + seat) % 2 == 0 else reversed(ARMS))}
                for i, seed in enumerate(SEEDS) for seat in (0, 1)]
    return dict(schema='master3s2-paired-cargo-defense-development/1', seeds=list(SEEDS), arms=list(ARMS),
                schedule=schedule, physical_boards=boards, files=files, platform=previous['platform'],
                examiner=previous['examiner'], registration_class='development', scope=previous['scope'], outcomes=previous['outcomes'])


def check(digest):
    require(sha(PLAN) == digest and read(PLAN) == contract(), 'Registered protocol or source bytes changed')
    return read(PLAN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'guard', 'exposure', 'freeze', 'check', 'list', 'play', 'verify', 'observations', 'settle'))
    parser.add_argument('--matrix-sha')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--seat', type=int, choices=(0, 1))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--image')
    parser.add_argument('--run', type=int)
    parser.add_argument('--commit')
    parser.add_argument('--archives', type=Path, nargs='+')
    args = parser.parse_args()
    if args.mode in ('prepare', 'guard', 'exposure'):
        sparse = load('cargo_defense_sparse', BASE / 'workflow_cli.py')
        sparse.HERE, sparse.SUITE, sparse.EXPOSURE_STEP, sparse.ARMS = HERE, WORKFLOW, EXPOSURE_STEP, ARMS
        if args.mode == 'prepare':
            sparse.prepare(args.matrix_sha)
        elif args.mode == 'guard':
            sparse.prior().guard(args.seed, args.seat)
        else:
            sparse.prior().exposure(args.out, args.seed, args.seat)
        return 0

    # The initial sparse checkout contains just this adapter and the shared bootstrap.
    global runner, play, sha, require, read, timing, verify_block
    runner = load('cargo_defense_cgroup_runner', BASE / 'matrix.py')
    play, sha, require, read = runner.play, runner.sha, runner.require, runner.read
    timing, verify_block = runner.timing, runner.verify_block
    # Both the public observation replayer and the settlement import the active matrix.
    sys.modules['matrix'] = sys.modules[__name__]
    runner.HERE, runner.PLAN, runner.SEEDS, runner.ARMS, runner.check = HERE, PLAN, SEEDS, ARMS, check
    if args.mode == 'freeze':
        expected = contract()
        if PLAN.exists():
            require(read(PLAN) == expected, 'Different existing freeze')
        else:
            play.live.save(PLAN, expected)
        print(sha(PLAN))
        return 0
    plan = check(args.matrix_sha)
    if args.mode == 'check':
        return 0
    if args.mode == 'list':
        print(json.dumps({'include': [{'seed': x['seed'], 'seat': x['seat']} for x in plan['schedule']]}))
        return 0
    if args.mode == 'settle':
        # The previous settlement also verifies that both archived public views reconstruct exactly.
        settler = load('cargo_defense_paired_settlement', PREVIOUS / 'settle.py')
        settler.audit.CONTRASTS = (('defense', 'parent'),)
        result = settler.audit.merge(args.run, args.commit, args.archives, args.out, args.matrix_sha)
        print(json.dumps(result['results'], indent=2))
        return int(not result['records_reconciled'])
    require(args.seed in SEEDS and args.seat in (0, 1) and args.out, 'Missing registered ticket or output')
    if args.mode == 'play':
        return runner.run(args.matrix_sha, args.seed, args.seat, args.out, args.image)
    if args.mode == 'verify':
        result = verify_block(args.out, args.seed, args.seat)
        print(json.dumps(result, allow_nan=False))
        return int(bool(result['verification_errors']))
    views = load('cargo_defense_public_views', STRATEGY / 'native_action_onpolicy_intake/observations.py')
    for arm in ARMS:
        folder = args.out / arm
        if (folder / 'turns.jsonl').exists():
            (folder / 'observations.jsonl.gz').write_bytes(views.recover(folder))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
