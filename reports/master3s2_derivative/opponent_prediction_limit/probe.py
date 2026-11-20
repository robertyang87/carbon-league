"""Isolate the effect of perfect first opponent commands on archived roots."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ROOT = FAMILY.parents[1]
PARENT = FAMILY / 'ladder/prediction_feedback/package/candidate.py'

sys.path.insert(0, str(FAMILY / 'modal_root_opponent'))
from predictions import actual, sha
sys.path.pop(0)

PROTOCOL_SHA = '0dddb84fc8007e2ea7762ee4571d7b82ba4df0ada962748fc9c0ecb3c1a89986'


def run(cpu):
    os.sched_setaffinity(0, {cpu})
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CARBON_DEBUG'] = '1'
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(FAMILY / 'root_rotation_policy/native'))
    from probe import each_history, equal_work_fn
    sys.path.insert(0, str(FAMILY / 'opponent_policy/search_probe'))
    from full_search_check import root_call

    raw = PARENT.read_bytes()
    registration = (FAMILY / 'modal_root_opponent/protocol.json').read_bytes()
    assert sha(registration) == PROTOCOL_SHA
    protocol = json.loads(registration)
    assert sha(raw) == protocol['packages'][0]
    mod = types.ModuleType('diagnostic_parent')
    exec(compile(raw, str(PARENT), 'exec'), mod.__dict__)
    assert mod._net() is not None and mod._SEARCH_OK and mod._TEMP == 1 and mod._S_ABORT == .58
    control = equal_work_fn()(mod, raw)
    native_search, native_sample = mod._search, mod._sample
    rules_file = ROOT / 'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json'
    cfg = {key: field['default'] for key, field in json.loads(rules_file.read_bytes())['configuration'].items()
           if isinstance(field, dict) and 'default' in field}
    cfg['episodeSteps'] = 300

    work_keys = ('candidates', 'depth', 'rollouts')
    matching_keys = (*work_keys, 'menu', 'act0', 'sorted_depths')
    replay_keys = (*matching_keys, 'issued', 'depth_by_arm', 'leaf_scores', 'opponent_samples', 'best', 'native_aborted')
    initial_cost = list(mod._S_COST)
    rows = []
    for first, last in zip(protocol['tickets'][::2], protocol['tickets'][1::2], strict=True):
        assert (first['archive'], first['seat']) == (last['archive'], last['seat'])
        targets = {first['step']: first, last['step']: last}
        seat = first['seat']
        tracker, previous = mod.BeliefTracker(mod._Bel, seat), None
        for obs, issued, checksum in each_history(first):
            step = obs['step']
            if step + 1 in targets:
                seed = targets[step + 1]['rng'] ^ 0x12F191AC
                cost = root_call(mod, tracker, previous, obs, cfg, seed, initial_cost)['ending_costs']
            if step in targets:
                ticket = targets[step]
                seed = ticket['rng']
                reference = root_call(mod, tracker, previous, obs, cfg, seed, cost)
                # Read the archived command after evaluating the natural reference.
                enemy, _, _ = actual(ticket)

                def actual_root_sample(players, player, lw, lb, rng, turn, fixed=None):
                    if players is obs['players'] and player == 1 - seat and turn == step:
                        assert fixed is None
                        return enemy.copy()
                    return native_sample(players, player, lw, lb, rng, turn, fixed)

                answers = {'sampled': [], 'actual': []}
                order = ('sampled', 'actual', 'actual', 'sampled') if len(rows) % 2 == 0 else (
                    'actual', 'sampled', 'sampled', 'actual')
                try:
                    mod._search = control
                    mod._response_cap, mod._response_depth, mod._S_MMAX = (reference[key] for key in work_keys)
                    for arm in order:
                        mod._sample = actual_root_sample if arm == 'actual' else native_sample
                        answers[arm].append(root_call(mod, tracker, previous, obs, cfg, seed, cost))
                finally:
                    mod._search, mod._sample, mod._S_MMAX = native_search, native_sample, mod._S_M

                recovered = all(ans[key] == reference[key] for ans in answers['sampled'] for key in replay_keys)
                intervened = all(ans['opponent_samples'] == [enemy] for ans in answers['actual'])
                matched = all(ans[key] == reference[key] for ans in answers['actual'] for key in matching_keys)
                _, repeated = answers['actual']
                repeatable = all(repeated[key] == answers['actual'][0][key] for key in replay_keys)
                valid = bool(reference['depth'] and recovered and intervened and matched and repeatable)
                changed = valid and answers['actual'][0]['issued'] != reference['issued']
                rows.append(dict(ticket=ticket, public_sha256=checksum, cost=cost, archived_enemy_order=enemy,
                                 reference=reference, answers=answers, recovered=recovered, intervened=intervened,
                                 matched=matched, repeatable=repeatable, valid=valid, changed=changed))
                print(ticket['seed'], seat, step, valid, changed, flush=True)
            tracker.step(obs['players'], step, obs['carbon'])
            tracker.set_actions(obs['players'], issued)
            previous = obs['players']
            if step == last['step']:
                break

    assert len(rows) == len(protocol['tickets']) == 16
    result = dict(protocol_sha256=PROTOCOL_SHA, parent=sha(raw), rows=rows)
    output = HERE / 'roots.json.gz'
    tmp = output.with_suffix('.partial')
    tmp.write_bytes(gzip.compress(json.dumps(result, allow_nan=False).encode() + b'\n', mtime=0))
    tmp.replace(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, required=True)
    run(parser.parse_args().cpu)
