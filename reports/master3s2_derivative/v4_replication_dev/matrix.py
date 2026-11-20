"""Replicate the frozen online V4 and V2 on four new maps against the official original."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ROOT = FAMILY.parents[1]
TEMPLATE = FAMILY / 'identity_replication_dev/matrix.py'
FIRST = FAMILY / 'nn_policy_development/protocol.json'
FIRST_SHA = 'bfe4589735ad6e335e1f649cb8222951cf6e86e6afd5299d2466dd37b427be71'
ORIGINAL = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
V2 = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'
V4 = 'cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a'
SEEDS = (621758610, 621758611, 621758612, 621758613)
WORKFLOW = 'master3s2-v4-replication-dev.yml'
EXPOSE = 'Play online V4 and V2 independently against original Master3s2'
spec = importlib.util.spec_from_file_location('four_map_original_comparison', TEMPLATE)
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)


def contract():
    runner = rep.paired
    require = runner.require
    require(runner.sha(FIRST) == FIRST_SHA, 'The first V4 comparison contract changed')
    old = runner.read(FIRST)
    original_registration = FAMILY / 'v3_release/development/protocol.json'
    require(runner.sha(original_registration) == old['files']['previous_v3_protocol']['sha256'],
            'Original referee registration changed')
    settings = runner.read(original_registration)
    shared = ('official_helpers', 'rules', 'environment', 'ladder_runner', 'worker', 'container',
              'base_registration', 'original_game_configuration', 'current_guide', 'match_runner',
              'replayer', 'native_timing_audit', 'previous_exposure_guard', 'registered_runner',
              'registered_settler', 'registered_sparse', 'observation_replay', 'new_identity', 'training_protocol')
    files = {name: old['files'][name] for name in shared}
    files.update(opponent=old['files']['original'], challenger=old['files']['new'],
                 challenger_weights=old['files']['trained_weights'], parent=old['files']['v2'],
                 parent_weights=old['files']['v2'], paired_settler=settings['files']['paired_settler'])
    additional = {'four_map_adapter': TEMPLATE, 'two_map_adapter': rep.OLD / 'matrix.py',
                  'paired_adapter': rep.initial.SOURCE / 'matrix.py', 'matrix': HERE / 'matrix.py',
                  'workflow': ROOT / '.github/workflows' / WORKFLOW, 'clearance_code': HERE / 'clearance.py',
                  'clearance': HERE / 'clearance.json', 'first_screen': FIRST,
                  'prior_reference': original_registration, 'live_v4_receipt': FAMILY / 'ladder/ds-pro-v4.json'}
    files.update({key: runner.runner.record(path) for key, path in additional.items()})
    require(all(runner.runner.record(ROOT / item['path']) == item for item in files.values()),
            'A referee component or competing package differs from its frozen bytes')
    live = runner.read(additional['live_v4_receipt'])
    require(live['verified_in_owner_list'] and live['version'] == 'ds-pro-v4'
            and [files[k]['sha256'] for k in ('challenger', 'parent', 'opponent')] == [V4, V2, ORIGINAL]
            and live['candidate_sha256'] == V4, 'Incorrect ladder V4, V2 or original source')
    cleared = runner.read(additional['clearance'])
    boards = cleared['by_seed']
    require(cleared['passed'] and set(map(int, boards)) == set(SEEDS) and len(set(boards.values())) == 4
            and not set(boards.values()).intersection(old['physical_boards'].values()) and
            not cleared['seed_intersections'] and not cleared['physical_intersections'] and
            cleared['verifier_sha256'] == files['clearance_code']['sha256'], 'Map clearance differs')
    tickets = runner.read(ROOT / files['training_protocol']['path'])['tickets']
    require(not any(row['seed'] in SEEDS or row['map'] in boards.values() for row in tickets),
            'Training or model-selection map in replication')
    stages = [dict(seed=seed, seat=seat, order=list(rep.ARMS if (i + seat) % 2 == 0 else reversed(rep.ARMS)))
              for i, seed in enumerate(SEEDS) for seat in (0, 1)]
    return dict(schema='ds-pro-v4-replication/1', registration_class='development', arms=list(rep.ARMS),
                seeds=list(SEEDS), physical_boards=boards, files=files, schedule=stages,
                platform=settings['platform'], examiner=settings['examiner'],
                scope='Both unchanged online packages play the unmodified original once per seat on each previously unused map. Two independent 1.5-core, 512-MiB agents, same host per map and seat, alternating match order, native search and natural draws. One attempt, no replacement, all 16 tickets retained.',
                outcomes='V4 and V2 wins/8 and official mean margins, map and seat splits, matched changes with each of four maps equally weighted. Development selection and the other two-map study remain separate. Final 100 reserved maps untouched.')


if __name__ == '__main__':
    rep.HERE, rep.SEEDS, rep.WORKFLOW, rep.EXPOSE, rep.contract = HERE, SEEDS, WORKFLOW, EXPOSE, contract
    raise SystemExit(rep.main())
