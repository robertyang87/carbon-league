"""Exercise the finalist check with live clocks on the frozen public histories."""
import argparse
import gzip
import inspect
import json
import os
from pathlib import Path
import sys
import types

from build import FAMILY, HERE, PARENT, PARENT_SHA, build, sha

ROOT = FAMILY.parents[1]
LOCK = '0dddb84fc8007e2ea7762ee4571d7b82ba4df0ada962748fc9c0ecb3c1a89986'


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
    import full_search_check

    namespace = {}
    tracer_source = inspect.getsource(full_search_check.root_call)
    assertion = 'len(enemy_samples) <= 1'
    assert tracer_source.count(assertion) == 1
    exec(tracer_source.replace(assertion, 'len(enemy_samples) <= 2'), vars(full_search_check), namespace)
    trace = namespace['root_call']
    candidate = build()
    protocol_raw = (FAMILY / 'modal_root_opponent/protocol.json').read_bytes()
    assert sha(protocol_raw) == LOCK
    tickets = json.loads(protocol_raw)['tickets']
    modules, original, controlled = {}, {}, {}
    for name, path, digest in (('parent', PARENT, PARENT_SHA), ('check', HERE / 'candidate.py', candidate)):
        raw = path.read_bytes()
        assert sha(raw) == digest
        mod = types.ModuleType(name)
        exec(compile(raw, str(path), 'exec'), mod.__dict__)
        assert mod._net() is not None and mod._S_ABORT == .58 and mod._TEMP == 1 and mod._SEARCH_OK
        modules[name], original[name], controlled[name] = mod, mod._search, equal_work_fn()(mod, raw)

    rules = ROOT / 'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json'
    cfg = {k: v['default'] for k, v in json.loads(rules.read_bytes())['configuration'].items()
           if isinstance(v, dict) and 'default' in v}
    cfg['episodeSteps'] = 300
    parent = modules['parent']
    opening = parent._S_COST[:]
    match = ('candidates', 'depth', 'rollouts', 'depth_by_arm', 'menu', 'act0', 'leaf_scores')
    out = []
    for first, last in zip(tickets[::2], tickets[1::2], strict=True):
        assert (first['archive'], first['seat']) == (last['archive'], last['seat'])
        targets = {first['step']: first, last['step']: last}
        seat = first['seat']
        tracker, previous = parent.BeliefTracker(parent._Bel, seat), None
        for obs, issued, checksum in each_history(first):
            step = obs['step']
            if step + 1 in targets:
                seed = targets[step + 1]['rng'] ^ 0x12F191AC
                cost = trace(parent, tracker, previous, obs, cfg, seed, opening)['ending_costs']
            if step in targets:
                ticket = targets[step]
                seed = ticket['rng']
                sequence = ('parent', 'check', 'check', 'parent') if len(out) % 2 == 0 else (
                    'check', 'parent', 'parent', 'check')
                results = {}
                for mode in ('native', 'matched_work'):
                    arms = {name: [] for name in modules}
                    for name in sequence:
                        mod = modules[name]
                        if mode == 'matched_work':
                            reference = results['native']['parent'][0]
                            mod._response_cap = reference['candidates']
                            mod._response_depth = reference['depth']
                            mod._search = controlled[name]
                        try:
                            mod._S_STATS.pop('response_check', None)
                            record = trace(mod, tracker, previous, obs, cfg, seed, cost)
                        finally:
                            mod._search = original[name]
                        record['check'] = mod._S_STATS.get('response_check')
                        if record['check'] is not None:
                            assert len(record['opponent_samples']) == 2
                            assert record['opponent_samples'][0] != record['opponent_samples'][1]
                        arms[name].append(record)
                    results[mode] = arms
                a = results['native']['parent'][0]
                matched = all(x[k] == a[k] for group in results['matched_work'].values() for x in group for k in match)
                matched = matched and all(x['opponent_samples'][:1] == a['opponent_samples'][:1]
                                          for group in results['matched_work'].values() for x in group)
                stable = matched and all(x['issued'] == a['issued']
                                         for group in results.values() for x in group['parent'])
                observed_veto = all(x['check'] is not None and x['check']['veto']
                                    and x['issued'] == a['act0'] != a['issued']
                                    for group in results.values() for x in group['check'])
                out.append(dict(ticket=ticket, public_sha256=checksum, cost=cost, arms=results,
                                matched=matched, stable_veto=stable and observed_veto))
                print(ticket['seed'], seat, step, matched, stable and observed_veto, flush=True)
            tracker.step(obs['players'], step, obs['carbon'])
            tracker.set_actions(obs['players'], issued)
            previous = obs['players']
            if step == last['step']:
                break
    assert len(out) == 16
    data = dict(protocol_sha256=LOCK, packages=[PARENT_SHA, candidate], rows=out)
    (HERE / 'results.json.gz').write_bytes(gzip.compress(json.dumps(data, allow_nan=False).encode(), mtime=0))


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--cpu', type=int, required=True)
    run(cli.parse_args().cpu)
