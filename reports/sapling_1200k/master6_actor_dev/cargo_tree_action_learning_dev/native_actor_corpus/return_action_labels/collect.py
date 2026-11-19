"""Collect single-command paired labels from locked historic states on one CPU."""
import argparse
import base64
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent
ROOT = next(p for p in HERE.parents if (p / '.git').exists())
PARENT = '8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac'
ORIGINAL = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'


def native_api():
    spec = importlib.util.spec_from_file_location('return_window_runner', CORPUS / 'original_response/run.py')
    windows = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(windows)
    return windows, windows.kernel()


def atomic(path, payload):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as fp:
        fp.write(payload)
        fp.flush()
        os.fsync(fp.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def suspend(k, original):
    snapshot = k.capture(original)
    return dict(snapshot_b64=base64.b64encode(snapshot).decode('ascii'),
                snapshot_sha=k.digest(snapshot), state_sha=k.sig(original))


def resume(k, original, state):
    snapshot = base64.b64decode(state['snapshot_b64'], validate=True)
    assert k.digest(snapshot) == state['snapshot_sha']
    k.restore(original, snapshot)
    assert k.sig(original) == state['state_sha']


class Progress:
    """Durably bind a unique timed draw to evidence before permitting its successor."""
    def __init__(self, k, folder, name, identity):
        self.k, self.identity = k, identity
        self.checkpoint = folder / (name + '.checkpoint.json.gz')
        self.in_flight = folder / (name + '.inflight.json')
        self.saved = k.R.load(self.checkpoint) if self.checkpoint.exists() else None
        if self.saved is not None:
            assert self.saved['identity'] == identity
        if self.in_flight.exists():
            marker = json.loads(self.in_flight.read_bytes())
            if not self.saved or marker != [identity, self.saved['step']]:
                raise RuntimeError(f'Interrupted original call has no durable return at {self.in_flight}. Its draw is unresolved.')
            self.in_flight.unlink()

    def starting(self, t):
        if self.in_flight.exists():
            raise RuntimeError(f'An original draw is already in progress at {self.in_flight}')
        atomic(self.in_flight, self.k.compact([self.identity, t]))

    def save(self, t, original, data):
        payload = dict(identity=self.identity, step=t, state=suspend(self.k, original), data=data)
        atomic(self.checkpoint, gzip.compress(self.k.compact(payload), compresslevel=1, mtime=0))
        self.in_flight.unlink(missing_ok=True)

    def clear(self):
        self.checkpoint.unlink(missing_ok=True)


def observe(k, env, seat, expected):
    own, enemy = k.public(env, seat), k.public(env, 1-seat)
    assert not env.done and own['step'] == enemy['step'] == expected
    return own, enemy, [k.digest(k.compact(o)) for o in (own, enemy)]


def clock(k, original, calls):
    return dict(calls=calls, summary=k.R.native_clock.summarize(calls, original._S_ABORT, original._S_D2) if calls else None)


def draw(k, original, recorder, native, obs, cfg):
    expected = len(recorder.calls) + 1
    command = k.R.live.clean(native(obs, cfg))
    assert len(recorder.calls) == expected and not recorder.errors and recorder.calls[-1]['step'] == obs['step']
    return command, recorder.calls[-1]


def prepare_shared(w, k, original, host, raw, roots, packs, folder, identity):
    path = folder / 'shared.json.gz'
    if path.exists():
        stored = k.R.load(path)
        assert stored['identity'] == identity
        return stored
    cfg, seat, last = raw['game_configuration'], identity['seat'], roots[-1]['t']
    targets = {r['t']: r for r in roots}
    prefix = raw['actions'][:last]
    assert len(prefix) == last and all(k.digest(k.compact(prefix[:r['t']])) == r['prefix_sha256'] for r in roots)
    actors = w.models(k, packs, 'archive')
    env = k.checked_env(raw)
    journal = Progress(k, folder, 'prefix', identity)
    data = dict(trace=[], calls=[], banks={}, host=host)
    if journal.saved:
        data = journal.saved['data']
        for t, row in enumerate(data['trace']):
            own, _, hashes = observe(k, env, seat, t)
            assert hashes == [row['own_sha'], row['enemy_sha']]
            commands, _ = w.commands(k, actors, own, cfg)
            assert all(row[a] == cmd for a, cmd in commands.items())
            if t < last:
                env.step(prefix[t])
        resume(k, original, journal.saved['state'])
    recorder = k.R.native_clock.Recorder(original)
    native = recorder.wrap(original.agent)
    for t in range(len(data['trace']), last + 1):
        own, enemy, hashes = observe(k, env, seat, t)
        proposed, timings = w.commands(k, actors, own, cfg)
        case = targets.get(t)
        if case:
            assert case['public_sha256'] == hashes[0] and all(case[a] == cmd for a, cmd in proposed.items())
            assert proposed['parent'] != proposed['student']
        journal.starting(t)
        enemy_command, call = draw(k, original, recorder, native, enemy, cfg)
        data['calls'].append(call)
        if case:
            # This state still contains the naturally issued root move, before retargeting the historical prefix.
            bank = dict(identity=identity, case=case, t=t, enemy=enemy_command, own_sha=hashes[0], enemy_sha=hashes[1],
                        prefix_sha=case['prefix_sha256'], **suspend(k, original))
            data['banks'][case['id']] = bank
        archived = k.R.live.clean(raw['actions'][t][1-seat]) if t < last else None
        data['trace'].append(dict(t=t, **proposed, own_sha=hashes[0], enemy_sha=hashes[1],
                                  original_proposed=enemy_command, archived_enemy=archived, own_public_seconds=timings))
        if t < last:
            original._ST['tr'].set_actions(enemy['players'], archived)
            env.step(prefix[t])
        journal.save(t, original, data)
    assert [r['step'] for r in data['calls']] == list(range(last + 1))
    for case in roots:
        root_folder = folder / case['id']
        root_folder.mkdir(exist_ok=True)
        w.save(k, root_folder / 'root.json.gz', data['banks'][case['id']])
    shared = dict(identity=identity, seed=identity['seed'], seat=seat, terrain=identity['terrain'],
                  prefix=prefix, trace=data['trace'], clock=clock(k, original, data['calls']), cpu=identity['cpu'],
                  host=data['host'], host_at_end=k.R.native_clock.host_snapshot(original))
    k.R.native_clock.validate_host(shared['host_at_end'], [identity['cpu']])
    w.save(k, path, shared)
    journal.clear()
    print('Durable common original prefix', identity['seed'], last, len(roots), flush=True)
    return shared


def branch(w, k, original, raw, common, root, packs, folder, arm, shared_sha):
    identity, case, t0, seat = common['identity'], root['case'], root['t'], common['seat']
    assert root['identity'] == identity and common['trace'][t0]['original_proposed'] == root['enemy']
    root_sha = k.R.sha(folder / 'root.json.gz')
    out = folder / (arm+'.json.gz')
    if out.exists():
        d = k.R.load(out)
        assert (d['identity'],d['arm'],d['root_sha'],d['shared_sha']) == (identity,arm,root_sha,shared_sha)
        assert d['tape_sha'] == k.digest(k.compact(d['tape']))
        return
    cfg = raw['game_configuration']
    env = k.checked_env(raw)
    parent = k.R.live.module(packs['parent'], 'return_continuation_'+arm)
    for t, joint in enumerate(common['prefix'][:t0]):
        own, _, hashes = observe(k, env, seat, t)
        row = common['trace'][t]
        assert hashes == [row['own_sha'],row['enemy_sha']] and k.own_call(parent, own, cfg)[0] == row['parent']
        env.step(joint)
    journal = Progress(k, folder, arm, dict(**identity, root_sha=root_sha, arm=arm))
    data = dict(tape=[], clock=[], observations=[], original_orders=[], own_wall=[])
    if journal.saved:
        data = journal.saved['data']
        for i, joint in enumerate(data['tape']):
            own, _, hashes = observe(k, env, seat, t0+i)
            cmd, _ = k.own_call(parent, own, cfg)
            assert hashes == data['observations'][i] and cmd == (case['parent'] if i == 0 else joint[seat])
            env.step(joint)
        resume(k, original, journal.saved['state'])
    else:
        resume(k, original, root)
    recorder = k.R.native_clock.Recorder(original)
    native = recorder.wrap(original.agent)
    data.setdefault('host_at_start', recorder.start)
    while not env.done:
        t = t0 + len(data['tape'])
        own, enemy, hashes = observe(k, env, seat, t)
        cmd, wall = k.own_call(parent, own, cfg)
        if t == t0:
            assert hashes == [root['own_sha'],root['enemy_sha']] and cmd == case['parent']
            cmd, enemy_cmd = case[arm], root['enemy']
            wall = common['trace'][t0]['own_public_seconds'][arm]
            resume(k, original, root)
        else:
            journal.starting(t)
            enemy_cmd, call = draw(k, original, recorder, native, enemy, cfg)
            data['clock'].append(call)
        joint = k.physical(cmd, enemy_cmd, seat)
        data['tape'].append(joint)
        data['observations'].append(hashes)
        data['original_orders'].append(enemy_cmd)
        data['own_wall'].append(wall)
        env.step(joint)
        journal.save(t, original, data)
    scores, cash, status = k.score(env)
    end = k.R.native_clock.host_snapshot(original)
    k.R.native_clock.validate_host(end,[identity['cpu']])
    tape = data['tape']
    assert [c['step'] for c in data['clock']] == list(range(t0+1,t0+len(tape)))
    assert all(en == both[1-seat] for en,both in zip(data['original_orders'], tape, strict=True))
    record = dict(identity=identity, arm=arm, case_id=case['id'], seat=seat, seed=case['seed'], root_t=t0,
                  root_joint=tape[0], tape=tape, tape_sha=k.digest(k.compact(tape)), clock=clock(k,original,data['clock']),
                  own_public_seconds=data['own_wall'], original_orders=data['original_orders'],
                  suffix_observations=data['observations'], host_at_start=data['host_at_start'], host_at_end=end,
                  status=status, physical_cash=cash, official_scores=scores, official_margin=scores[seat]-scores[1-seat],
                  state_sha=root['state_sha'], snapshot_sha=root['snapshot_sha'], root_sha=root_sha, shared_sha=shared_sha)
    w.save(k,out,record)
    journal.clear()
    print('Sealed', case['id'], arm, len(tape), 'official transitions', flush=True)


def run(roots_sha, cpu, index=None):
    assert cpu in (0, 1) and (index is None or index % 2 == cpu)
    os.sched_setaffinity(0, {cpu})
    w,k = native_api()
    protocol, registry = HERE/'protocol.json', HERE/'roots.json'
    plan, selection = k.R.load(protocol), k.R.load(registry)
    assert __debug__ and k.R.sha(registry) == roots_sha and k.R.sha(protocol) == selection['protocol_sha256']
    assert all(k.R.sha(ROOT / item['path']) == item['sha256'] for item in plan['inputs'].values())
    assert k.R.sha(k.ORIGINAL) == plan['original_sha256'] == ORIGINAL
    packs = {name: ROOT/p['path'] for name,p in plan['packages'].items()}
    assert set(packs)=={'parent','student'} and all(k.R.sha(p)==plan['packages'][name]['sha256'] for name,p in packs.items())
    assert k.R.sha(packs['parent']) == PARENT and k.R.sha(packs['student']) != PARENT
    cards = k.R.load(CORPUS/'sources.json')['cards']
    maps = plan['maps']; roots = selection['roots']
    assert len(maps)==4 and len({c['terrain'] for c in maps})==4 and len({r['id'] for r in roots})==len(roots)
    assert {r['seed'] for r in roots} <= {c['seed'] for c in maps}
    with (HERE/f'collection_cpu{cpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for i,card in enumerate(maps):
            if i % 2 != cpu or (index is not None and i != index):
                continue
            group = sorted([r for r in roots if r['seed']==card['seed']], key=lambda r:r['t'])
            if not group:
                print('Zero eligible roots', card['seed'], flush=True)
                continue
            source,receipt = ROOT/card['source'], ROOT/card['receipt']
            assert k.R.sha(source)==card['source_sha256'] and k.R.sha(receipt)==card['receipt_sha256']
            assert any(c['split']=='train' and c['sha256']==card['source_sha256'] and c['terrain']==card['terrain'] and c['seed']==card['seed'] and c['seat']==card['seat'] for c in cards)
            raw=k.R.load(source)
            assert raw['seed']==card['seed'] and raw['candidate_seat']==card['seat'] and raw['initial_terrain_u8_sha256']==card['terrain'] and not raw['errors']
            assert all(all(r[f]==card[f] for f in ('source','source_sha256','seed','seat','terrain')) and 210<=r['t']<=265 and sorted(r['arm_order'])==['parent','student'] for r in group)
            assert len(set(r['t'] for r in group))==len(group)<=2
            folder=HERE/'native'/f"{card['seed']}_{card['seat']}"
            folder.mkdir(parents=True, exist_ok=True)
            identity=dict(protocol=k.R.sha(protocol),roots=roots_sha,runner=k.R.sha(Path(__file__)),kernel=k.R.sha(w.KERNEL),
                          window_kernel=k.R.sha(Path(w.__file__)),original=ORIGINAL,packages={n:k.R.sha(p) for n,p in packs.items()},
                          source=card['source_sha256'],source_receipt=card['receipt_sha256'],seed=card['seed'],seat=card['seat'],terrain=card['terrain'],cpu=cpu)
            original,host=k.load_master(cpu)
            common=prepare_shared(w,k,original,host,raw,group,packs,folder,identity)
            shared_sha=k.R.sha(folder/'shared.json.gz')
            for case in group:
                directory=folder/case['id'];root=k.R.load(directory/'root.json.gz')
                assert root['case']==case
                for arm in case['arm_order']:
                    branch(w,k,original,raw,common,root,packs,directory,arm,shared_sha)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots-sha',required=True,help='SHA-256 of the sealed roots.json')
    parser.add_argument('--cpu',type=int,choices=(0,1),required=True,help='Protocol map indices 0,2 run on CPU0. Indices 1,3 run on CPU1.')
    parser.add_argument('--index',type=int,choices=range(4),default=None,help='An optional fixed map index. Default collects both maps assigned to the CPU.')
    args=parser.parse_args()
    run(args.roots_sha,args.cpu,args.index)
