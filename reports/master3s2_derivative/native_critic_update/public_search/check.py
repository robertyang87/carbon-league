"""Compare two pinned packages on the same public histories, clock budget, and sampled actions."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import types
import zipfile

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parents[1]
ROOT = FAMILY.parents[1]
PARENT = FAMILY / 'advantage_policy_update/v2_bc.py'
CANDIDATE = HERE.parent / 'candidate.py'
DIGESTS = ('cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a',
           '04169552bb2aa670e98c9458dfa45b35756dff68a206935a142aa811cd4b71a9')
ORIGINAL = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
STEPS = (117, 247)
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def registration():
    tickets = []
    for path in sorted((FAMILY / 'v4_replication_dev/hosted').glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            raw = archive.read('challenger/registration.json')
        record = json.loads(raw)
        assert record['files']['candidate']['sha256'] == DIGESTS[0]
        assert record['files']['opponent']['sha256'] == ORIGINAL
        seed, seat = record['seed'], record['candidate_seat']
        assert seed in range(621758610, 621758614)
        for t in STEPS:
            rng = int(sha(f'critic-public-search/1/{seed}/{seat}/{t}'.encode())[:8], 16)
            tickets.append(dict(archive=str(path.relative_to(FAMILY)), archive_sha256=sha(path.read_bytes()),
                                arm='challenger', registration_sha256=sha(raw), seed=seed, seat=seat,
                                terrain=record['physical_map_sha256'], step=t, rng=rng))
    assert len(tickets) == len({(c['seed'], c['seat'], c['step']) for c in tickets}) == 16
    return dict(schema='native-critic-clock/1', packages=DIGESTS, tickets=tickets,
                primary='Both repeated native actions differ between arms, agree with same-work control, and both controls share enemy root commands and depth distribution.',
                scope='All two preselected steps, two seats on each of four held development boards. Alternate ABBA/BAAB. Forced-work results diagnostic.')


def execute(core):
    os.sched_setaffinity(0, {core})
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CARBON_DEBUG'] = '1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(FAMILY / 'root_rotation_policy/native'))
    from probe import each_history, equal_work_fn
    sys.path.insert(0, str(FAMILY / 'opponent_policy/search_probe'))
    from full_search_check import root_call

    prot = (HERE / 'protocol.json').read_bytes()
    tickets = json.loads(prot)['tickets']
    assert prot == encode(registration())
    mods, native, control = {}, {}, {}
    make_control = equal_work_fn()
    for name, path, expected in zip(('v4', 'critic'), (PARENT, CANDIDATE), DIGESTS):
        raw = path.read_bytes()
        assert sha(raw) == expected
        module = types.ModuleType(name)
        exec(compile(raw, name, 'exec'), module.__dict__)
        assert module._net() is not None and module._S_ABORT == .58 and module._TEMP == 1 and module._SEARCH_OK
        mods[name], native[name], control[name] = module, module._search, make_control(module, raw)
    parent = mods['v4']
    rules = json.loads((ROOT / 'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json').read_bytes())
    cfg = {k: v['default'] for k, v in rules['configuration'].items() if isinstance(v, dict) and 'default' in v}
    cfg['episodeSteps'] = 300
    opening = list(parent._S_COST)
    match_keys = ('act0', 'issued', 'menu', 'depth_by_arm', 'leaf_scores', 'opponent_samples', 'best', 'native_aborted')
    rows = []
    for a, b in zip(tickets[::2], tickets[1::2], strict=True):
        assert a['archive'] == b['archive'] and a['seat'] == b['seat']
        targets = {r['step']: r for r in (a, b)}
        tracker, previous = parent.BeliefTracker(parent._Bel, a['seat']), None
        for obs, issued, checksum in each_history(a):
            t = obs['step']
            if t + 1 in targets:
                warm = root_call(parent, tracker, previous, obs, cfg, targets[t+1]['rng'] ^ 0x12F191AC, opening)
            if t in targets:
                ticket = targets[t]
                order = ('v4', 'critic', 'critic', 'v4') if len(rows) % 2 == 0 else ('critic', 'v4', 'v4', 'critic')
                row = dict(ticket=ticket, public_sha256=checksum, cost=warm['ending_costs'], order=order)
                for mode in ('live', 'equal_work'):
                    results = {n: [] for n in mods}
                    for n in order:
                        module = mods[n]
                        if mode == 'equal_work':
                            module._response_cap, module._response_depth, module._S_MMAX = work
                            module._search = control[n]
                        try:
                            result = root_call(module, tracker, previous, obs, cfg, ticket['rng'], row['cost'])
                        finally:
                            module._search = native[n]
                            module._S_MMAX = module._S_M
                        results[n].append(result)
                    row[mode] = results
                    if mode == 'live':
                        reference = results['v4'][0]
                        work = tuple(reference[k] for k in ('candidates', 'depth', 'rollouts'))
                live, fixed = row['live'], row['equal_work']
                row['live_change'] = live['v4'][0]['issued'] != live['critic'][0]['issued']
                row['repeat'] = all(group[0]['issued'] == group[1]['issued'] for group in live.values())
                row['same_issued'] = all(q['issued'] == live[n][0]['issued'] for n in mods for q in fixed[n])
                row['recovered_parent'] = all(q[k] == reference[k] for q in fixed['v4'] for k in match_keys)
                row['same_work'] = bool(reference['depth'] and reference['opponent_samples']) and all(
                    q['sorted_depths'] == reference['sorted_depths'] and q['opponent_samples'] == reference['opponent_samples']
                    and tuple(q[k] for k in ('candidates', 'depth', 'rollouts')) == work
                    for group in fixed.values() for q in group)
                row['reliable_change'] = all(row[k] for k in ('live_change', 'repeat', 'same_issued', 'recovered_parent', 'same_work'))
                rows.append(row)
                print(ticket['seed'], ticket['seat'], t, *(row[k] for k in ('live_change', 'same_work', 'reliable_change')), flush=True)
            tracker.step(obs['players'], t, obs['carbon'])
            tracker.set_actions(obs['players'], issued)
            previous = obs['players']
            if t == STEPS[-1]:
                break
    assert len(rows) == 16
    output = dict(protocol_sha256=sha(prot), clock_tests=64, same_work_tests=64, rows=rows)
    (HERE / 'results.json.gz').write_bytes(gzip.compress(encode(output), mtime=0))


def encode(data):
    return (json.dumps(data, indent=2, allow_nan=False) + '\n').encode()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--core', type=int)
    args = parser.parse_args()
    if args.register:
        raw = encode(registration())
        path = HERE / 'protocol.json'
        assert not path.exists() or path.read_bytes() == raw
        path.write_bytes(raw)
        print(sha(raw))
    else:
        execute(args.core)
