"""Two fresh maps, both seats and both policies against the registered original."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
FAMILY = LANE.parent
PRIOR = FAMILY / 'ladder/cargo_defense/development'
REPO = next(p for p in HERE.parents if (p / '.git').exists())
PRIOR_SHA = '244d7cc3f531ab0806db3c859a651298cddff962aa63c81e63fc51cebe1fac6e'
WORKFLOW = 'master3s2-return-rank-development.yml'
SEEDS = (621758598, 621758599)
ARMS = ('parent', 'return_rank')
spec = importlib.util.spec_from_file_location('paired_development', PRIOR / 'matrix.py')
paired = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = paired
spec.loader.exec_module(paired)


def contract():
    previous = paired.read(PRIOR / 'protocol.json')
    require = paired.require
    require(paired.sha(PRIOR / 'protocol.json') == PRIOR_SHA, 'Different original settlement contract')
    keys = ('opponent', 'official_helpers', 'rules', 'environment', 'ladder_runner', 'worker', 'container',
            'base_registration', 'original_game_configuration', 'current_guide', 'match_runner', 'replayer',
            'native_timing_audit', 'previous_exposure_guard', 'registered_runner', 'registered_settler',
            'registered_sparse', 'observation_replay', 'parent', 'parent_weights', 'paired_settler')
    files = {k: previous['files'][k] for k in keys}
    require(all(paired.runner.record(REPO / r['path']) == r for r in files.values()), 'Changed examiner, parent or referee')
    pkg = LANE / 'package'
    paths = {'training_freeze': pkg / 'frozen_inputs.json', 'return_rank': pkg / 'candidate.py', 'return_rank_weights': LANE / 'weights.json',
             'model': LANE / 'model.py', 'builder': pkg / 'build.py', 'insertion': pkg / 'offer.py',
             'manifest': pkg / 'manifest.json', 'offer_protocol': pkg / 'protocol.json', 'fit_manifest': LANE / 'train_manifest.json',
             'held': LANE / 'offline.json', 'source': LANE / 'fit.py', 'clearance': HERE / 'clearance.json',
             'clearance_code': HERE / 'clearance.py', 'clearance_helper': PRIOR / 'clearance.py',
             'prior_reference': PRIOR / 'protocol.json', 'paired_adapter': PRIOR / 'matrix.py',
             'root_protocol': HERE / 'root_protocol.json', 'root_verifier': HERE / 'root_responses.py',
             'root_result': HERE / 'root_results.json.gz', 'workflow': REPO / '.github/workflows' / WORKFLOW,
             'matrix': HERE / 'matrix.py'}
    files.update({k: paired.runner.record(v) for k,v in paths.items()})
    build = paired.read(paths['manifest'])
    require(build['candidate_sha256'] == files['return_rank']['sha256'] and
            build['parent_sha256'] == files['parent']['sha256'] == paired.PARENT and files['opponent']['sha256'] == paired.ORIGINAL
            and build['bytes'] == files['return_rank']['bytes'] <= 10_000_000, 'Wrong candidate or reference')
    for k, field in [('return_rank_weights','weights_sha256'), ('insertion','integration_sha256'), ('model','model_sha256'),
                     ('offer_protocol','protocol_sha256')]:
        require(files[k]['sha256'] == build[field], 'Release identity disagrees for ' + k)
    frozen = paired.read(paths['training_freeze'])
    require(frozen['weights_sha256'] == build['weights_sha256'] == '8afb999bcf3a719399370a1ed508d1776234d9f9d80b97360fee7487e5384062'
            and frozen['model_sha256'] == build['model_sha256'], 'Model with repaired diagonal cargo-tax accounting required')
    offline = paired.read(paths['held'])
    require(offline['verdict'] == 'PASS' and offline['weight_sha256'] == build['weights_sha256'], 'Unqualified held ranking')
    probe = json.loads(gzip.decompress(paths['root_result'].read_bytes()))
    require(probe['candidate_sha256'] == build['candidate_sha256'] and probe['protocol_sha256'] == files['root_protocol']['sha256']
            and probe['complete'] and len(probe['rows']) == 24, 'Incomplete or stale root evidence')
    fit = paired.read(paths['fit_manifest'])
    roots = paired.read(paths['root_protocol'])
    census = paired.read(paths['clearance'])
    boards = census['by_seed']
    require(census['passed'] and set(map(int, boards)) == set(SEEDS) and len(set(boards.values())) == len(SEEDS)
            and not census['physical_intersections'] and not census['seed_intersections'] and census['boards'] == boards
            and census['verifier_sha256'] == files['clearance_code']['sha256']
            and census['source_census_runner_sha256'] == files['clearance_helper']['sha256'], 'Map clearance mismatch')
    for label, games in [('trained maps', fit['tickets']), ('old root probes', roots['roots'])]:
        require(all(r['seed'] not in SEEDS and r['terrain'] not in boards.values() for r in games), label + ' intersection')
    schedule = [{'seed': seed, 'seat': seat, 'order': list(ARMS if (n+seat)%2 == 0 else reversed(ARMS))}
                for n, seed in enumerate(SEEDS) for seat in (0,1)]
    return dict(schema='master3s2-return-ranking-development/1', seeds=list(SEEDS), arms=list(ARMS), schedule=schedule,
                physical_boards=boards, files=files, platform=previous['platform'], examiner=previous['examiner'],
                registration_class='development', scope=previous['scope'], outcomes=previous['outcomes'])


def main():
    paired.HERE, paired.PLAN = HERE, HERE / 'protocol.json'
    paired.SEEDS, paired.ARMS, paired.WORKFLOW = SEEDS, ARMS, WORKFLOW
    paired.EXPOSURE_STEP = 'Play parent and return-ranking candidate against original'
    paired.contract = contract
    if sys.argv[1:2] == ['settle']:
        # The archived four-block referee permits an explicit contrast for this two-arm tournament.
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('settle')
        parser.add_argument('--run', type=int, required=True)
        parser.add_argument('--commit', required=True)
        parser.add_argument('--matrix-sha', required=True)
        parser.add_argument('--out', type=Path, required=True)
        parser.add_argument('--archives', nargs='+', type=Path, required=True)
        args = parser.parse_args()
        sys.argv[1:] = ['check', '--matrix-sha', args.matrix_sha]
        paired.main()
        settler = paired.load('return_rank_settler', paired.PREVIOUS / 'settle.py')
        settler.audit.CONTRASTS = (('return_rank', 'parent'),)
        result = settler.audit.merge(args.run, args.commit, args.archives, args.out, args.matrix_sha)
        print(json.dumps(result['results'], indent=2))
        return int(not result['records_reconciled'])
    return paired.main()


if __name__ == '__main__':
    raise SystemExit(main())
