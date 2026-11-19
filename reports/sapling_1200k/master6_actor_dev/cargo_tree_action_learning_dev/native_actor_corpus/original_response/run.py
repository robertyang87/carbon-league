"""Run both preregistered public windows serially after a separately signed release."""
import argparse
import base64
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / '.git').exists())
EXP = ROOT / 'reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev'
KERNEL = EXP / 'learned_carbon_observation/native_response/run.py'
EXAMINER = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'


def kernel():
    spec = importlib.util.spec_from_file_location('native_window_kernel', KERNEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(k, path, value):
    k.blob_write(path, gzip.compress(k.compact(value), mtime=0))


def models(k, packs, tag):
    return {a: k.R.live.module(p, 'trunk_window_' + tag + '_' + a) for a, p in packs.items()}


def commands(k, actors, obs, cfg):
    result = {a: k.own_call(agent, obs, cfg) for a, agent in actors.items()}
    return {a: v[0] for a, v in result.items()}, {a: v[1] for a, v in result.items()}


def shared_root(k, original, host, case, packs, raw, folder, identity):
    bank_path = folder / 'shared.json.gz'
    if bank_path.exists():
        bank = k.R.load(bank_path)
        assert bank['identity'] == identity
    else:
        env = k.checked_env(raw)
        seat, t0, cfg = case['seat'], case['t0'], raw['game_configuration']
        actors = models(k, packs, 'prefix')
        recorder = k.R.native_clock.Recorder(original)
        native = recorder.wrap(original.agent)
        prefix, trace = raw['actions'][:t0], []
        assert len(prefix) == t0
        started = time.perf_counter()
        for t in range(t0 + 1):
            own, enemy = k.public(env, seat), k.public(env, 1 - seat)
            assert not env.done and own['step'] == enemy['step'] == t
            sampled = k.R.live.clean(native(enemy, cfg))
            emitted, timings = commands(k, actors, own, cfg)
            committed = sampled if t == t0 else prefix[t][1 - seat]
            obs_hashes = dict(own_sha=k.digest(k.compact(own)), enemy_sha=k.digest(k.compact(enemy)))
            trace.append(dict(t=t, **obs_hashes, **emitted, own_public_seconds=timings,
                              original_proposed=sampled, original_executed=committed))
            if t < t0:
                original._ST['tr'].set_actions(enemy['players'], committed)
                env.step(prefix[t])
        root = dict(t=t0, enemy=sampled, **emitted, **obs_hashes)
        state, snapshot = k.sig(original), k.capture(original)
        k.restore(original, snapshot)
        assert k.sig(original) == state
        clock = k.clock_result(recorder, original)
        assert [v['step'] for v in clock['calls']] == list(range(t0 + 1))
        bank = dict(identity=identity, case=case, root=root, prefix=prefix, trace=trace, host=host,
                    cpu=identity['cpu'], prefix_sha=k.digest(k.compact(prefix)), clock=clock,
                    state_sha=state, snapshot_sha=k.digest(snapshot), wall_s=time.perf_counter()-started,
                    snapshot_b64=base64.b64encode(snapshot).decode('ascii'))
        save(k, bank_path, bank)
    snapshot = base64.b64decode(bank['snapshot_b64'], validate=True)
    assert k.digest(snapshot) == bank['snapshot_sha']
    k.blob_write(folder / 'original_root.pickle.gz', gzip.compress(snapshot, mtime=0))
    return bank, snapshot


def rollout(k, original, case, packs, raw, folder, common, snapshot, arm):
    path = folder / (arm + '.json.gz')
    identity = common['identity']
    shared_sha = k.digest(k.compact(common))
    if path.exists():
        old = k.R.load(path)
        assert old['identity'] == identity and old['shared_sha'] == shared_sha and old['arm'] == arm
        assert old['tape_sha'] == k.digest(k.compact(old['tape']))
        return
    seat, t0, cfg = case['seat'], case['t0'], raw['game_configuration']
    assert case['t1_exclusive'] == t0 + 32
    env, actors = k.checked_env(raw), models(k, packs, arm)
    for t, archived in enumerate(common['prefix']):
        own = k.public(env, seat)
        emitted, _ = commands(k, actors, own, cfg)
        assert all(common['trace'][t][a] == cmd for a, cmd in emitted.items())
        assert k.digest(k.compact(own)) == common['trace'][t]['own_sha']
        env.step(archived)
    k.restore(original, snapshot)
    assert k.sig(original) == common['state_sha']
    recorder = k.R.native_clock.Recorder(original)
    native = recorder.wrap(original.agent)
    started = time.perf_counter()
    tape, comparisons, own_wall, observations = [], [], [], []
    while not env.done:
        t = t0 + len(tape)
        own, enemy = k.public(env, seat), k.public(env, 1-seat)
        assert t == own['step'] == enemy['step']
        hashes = [k.digest(k.compact(own)), k.digest(k.compact(enemy))]
        observations.append(hashes)
        if t == t0:
            assert hashes == [common['root']['own_sha'], common['root']['enemy_sha']]
            enemy_command = common['root']['enemy']
        else:
            enemy_command = k.R.live.clean(native(enemy, cfg))
        if t < case['t1_exclusive']:
            emitted, timings = commands(k, actors, own, cfg)
            if t == t0:
                assert all(common['root'][a] == cmd for a, cmd in emitted.items())
            command = emitted[arm]
            comparisons.append(dict(t=t, **emitted, executed=command, changed=emitted['parent'] != emitted['student'],
                                    worker_changed=k.agreements(own, emitted['parent'], emitted['student']), own_public_seconds=timings))
            own_wall.append(timings[arm])
        else:
            command, wall = k.own_call(actors['parent'], own, cfg)
            own_wall.append(wall)
        joint = k.physical(command, enemy_command, seat)
        tape.append(joint)
        env.step(joint)
    scores, cash, status = k.score(env)
    clock = k.clock_result(recorder, original)
    end = k.R.native_clock.host_snapshot(original)
    k.R.native_clock.validate_host(end, [identity['cpu']])
    assert [v['step'] for v in clock['calls']] == list(range(t0 + 1, t0 + len(tape)))
    record = dict(arm=arm, seed=case['seed'], seat=seat, root_t=t0, root_joint=tape[0], tape=tape,
                  clock=clock, tape_sha=k.digest(k.compact(tape)), identity=identity, shared_sha=shared_sha,
                  state_sha=common['state_sha'], snapshot_sha=common['snapshot_sha'], host_at_start=recorder.start,
                  host_at_end=end, status=status, physical_cash=cash, official_scores=scores,
                  official_margin=scores[seat]-scores[1-seat], comparisons=comparisons, own_public_seconds=own_wall,
                  suffix_observations=observations, wall_s=time.perf_counter()-started)
    save(k, path, record)
    print('Completed', case['seed'], arm, len(tape), 'decisions', flush=True)


def run(release, output, cpu, idx):
    from freeze import revalidate
    os.sched_setaffinity(0, {cpu})
    plan = revalidate(json.loads(release.read_bytes()))
    k = kernel()
    contract = k.R.load(HERE / 'protocol.json')
    assert __debug__ and k.R.sha(release) == contract['release_sha256']
    assert all(k.R.sha(ROOT / path) == sha for path, sha in contract['files'].items())
    assert k.R.sha(k.ORIGINAL) == EXAMINER == contract['original_sha256']
    packs = {a: ROOT / plan['packages'][p]['path'] for a, p in [('parent','parent'), ('student','full')]}
    output.mkdir(parents=True, exist_ok=True)
    with (output / f'native_execution_{idx}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        case = plan['cases'][idx]
        directory = output / 'native' / f"{case['seed']}_{case['seat']}"
        directory.mkdir(parents=True, exist_ok=True)
        identity = dict(protocol=k.R.sha(HERE / 'protocol.json'), release=k.R.sha(release), runner=k.R.sha(Path(__file__)), gate=k.R.sha(HERE/'freeze.py'),
                        kernel=k.R.sha(KERNEL), original=EXAMINER, cpu=cpu, source=case['source_sha256'],
                        packages={a:k.R.sha(p) for a,p in packs.items()}, seed=case['seed'], seat=case['seat'])
        original, host = k.load_master(cpu)
        source = ROOT / case['source']
        assert k.R.sha(source) == case['source_sha256']
        raw = k.R.load(source)
        assert raw['seed'] == case['seed'] and raw['candidate_seat'] == case['seat']
        common, snapshot = shared_root(k, original, host, case, packs, raw, directory, identity)
        for arm in case['arm_order']:
            rollout(k, original, case, packs, raw, directory, common, snapshot, arm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cpu', required=True, type=int)
    parser.add_argument('--case', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    run(args.freeze, args.out, args.cpu, args.case)
