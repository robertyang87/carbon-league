"""Compare the critic-only challenger and its parent against the complete original."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ROOT = FAMILY.parents[1]
SEEDS = tuple(range(621782436, 621782440))
WORKFLOW = 'master3s2-native-critic-dev.yml'
EXPOSE = 'Play native critic and V4 against original Master3s2'
TEMPLATE = FAMILY / 'identity_replication_dev/matrix.py'
PARENT = 'cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a'
CRITIC = '04169552bb2aa670e98c9458dfa45b35756dff68a206935a142aa811cd4b71a9'
ORIGINAL = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
spec = importlib.util.spec_from_file_location('four_map_official_comparison', TEMPLATE)
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)


def contract():
    runner = rep.paired
    previous = FAMILY / 'v4_replication_dev/protocol.json'
    old = runner.read(previous)
    shared = ('official_helpers', 'rules', 'environment', 'ladder_runner', 'worker', 'container',
              'base_registration', 'original_game_configuration', 'current_guide', 'match_runner',
              'replayer', 'native_timing_audit', 'previous_exposure_guard', 'registered_runner',
              'registered_settler', 'registered_sparse', 'observation_replay', 'new_identity',
              'training_protocol', 'paired_settler')
    files = {name:old['files'][name] for name in shared}
    files['opponent'] = old['files']['opponent']
    files['parent'] = old['files']['challenger']
    files['parent_weights'] = old['files']['challenger_weights']
    paths = {'challenger': FAMILY / 'native_critic_update/candidate.py',
             'challenger_weights': FAMILY / 'native_critic_update/weights.npz',
             'critic_training': FAMILY / 'native_critic_update/protocol.json',
             'prior_comparison': previous, 'four_map_adapter': TEMPLATE,
             'two_map_adapter': rep.OLD / 'matrix.py', 'paired_adapter': rep.initial.SOURCE / 'matrix.py',
             'matrix': Path(__file__).resolve(), 'workflow': ROOT / '.github/workflows' / WORKFLOW,
             'clearance_code': HERE / 'clearance.py', 'clearance': HERE / 'clearance.json'}
    files.update({name:runner.runner.record(p) for name,p in paths.items()})
    runner.require(all(runner.runner.record(ROOT / c['path']) == c for c in files.values()), 'Modified source bytes')
    runner.require([files[k]['sha256'] for k in ('parent','challenger','opponent')] == [PARENT,CRITIC,ORIGINAL],
                   'Unexpected package identity')
    census = runner.read(paths['clearance'])
    boards = census['by_seed']
    runner.require(census['passed'] and set(map(int, boards)) == set(SEEDS)
                   and census['verifier_sha256'] == files['clearance_code']['sha256']
                   and not census['seed_intersections'] and not census['physical_intersections']
                   and len(set(boards.values())) == 4 and not set(boards.values()) & set(old['physical_boards'].values()),
                   'Map clearance differs')
    learning = runner.read(paths['critic_training'])
    known_seeds = set(learning['excluded_seeds'])
    known_terrains = set(learning['excluded_terrain'])
    known_seeds.update(card['seed'] for card in learning['tickets'] if card['seed'] is not None)
    known_terrains.update(card['terrain'] for card in learning['tickets'])
    runner.require(not known_seeds.intersection(SEEDS) and not known_terrains.intersection(boards.values()),
                   'A training, validation or known action-pair board appears in the new comparison')
    schedule = [dict(seed=seed, seat=seat, order=list(rep.ARMS if (idx+seat)%2==0 else reversed(rep.ARMS)))
                for idx,seed in enumerate(SEEDS) for seat in (0,1)]
    return dict(schema='ds-pro-native-critic-development/1', registration_class='development', arms=list(rep.ARMS),
                seeds=list(SEEDS), physical_boards=boards, files=files, schedule=schedule,
                platform=old['platform'], examiner=old['examiner'],
                scope='Each package plays the unchanged original at both seats of four new development maps. Same host per map/seat pair, alternate order, native clocks and natural sampling. Retain every outcome; no replay replacement.',
                outcomes='Report both arms wins/8 and mean official margin with paired map-level effects. Formal 100-map exam remains sealed.')


if __name__ == '__main__':
    rep.HERE, rep.SEEDS, rep.WORKFLOW, rep.EXPOSE, rep.contract = HERE, SEEDS, WORKFLOW, EXPOSE, contract
    raise SystemExit(rep.main())
