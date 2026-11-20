"""Restore 8b4c root inputs from the acting seat's information, without reading suffixes."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
DERIVATIVE = HERE.parents[1]
CLOSE = DERIVATIVE / 'return_response_learning/close_games'
MODEL = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'
MENU = '3b737e03d58639eeca74303c8798b517172d69028ea9db3ac961669ec58b355a'
SNAPSHOTS = {
    621758592: '41639227a7f53ff67a5c5bd27617451e637d0815b8d6ae9eb7c5a800f6b4e87a',
    621758600: 'd51d2bbc53526c1cdea4da93bfba6cf0cd3dfef45183b18cdd3b1a90688153a1',
    621758602: 'b2238b5969dd73705980b2d612d1f95cd01ce3592966164cb842001829832e9e',
    621758603: '9ff468a17293164e38a406d5e764dca3fc702695c0dc2f3fce4cd55b8bdd955e',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(obs):
    return sha((json.dumps(obs, separators=(',', ':'), allow_nan=False) + '\n').encode())


def restore(ticket, model):
    """Return (obs, effective native cfg, estimated carbon, previous players, [A,B], native A−B)."""
    seed, seat, step = [ticket[k] for k in ('seed', 'seat', 'step')]
    assert ticket['group'] == 'close' and sha(Path(model.__file__).read_bytes()) == MODEL
    raw = (CLOSE / 'menus.json').read_bytes()
    assert sha(raw) == MENU == ticket['root_sha']
    row = json.loads(raw)['rows'][ticket['close_ordinal']]
    r = row['ticket']
    assert [r[k] for k in ('seed', 'seat', 'step')] == [seed, seat, step]
    assert r['terrain'] == ticket['terrain'] and r['sha256'] == ticket['source_sha']
    assert row['dep'] == [8, 8] and row['native_top_two'] and row['orders'] == [row['menu'][i] for i in row['indices']]
    arc = DERIVATIVE / r['path']
    assert sha(arc.read_bytes()) == r['sha256']
    tracker = model.BeliefTracker(model._Bel, seat)
    prev = None
    with zipfile.ZipFile(arc) as z, z.open(r['arm']+'/observations.jsonl.gz') as stream, gzip.GzipFile(fileobj=stream) as frames, z.open(r['arm']+'/turns.jsonl') as turns:
        for t in range(step + 1):
            frame = json.loads(next(frames))
            obs = frame['observations'][seat]
            assert obs['step'] == t and obs['player'] == seat and digest(obs) == frame['public_observation_sha256'][seat]
            est, known = tracker.step(obs['players'], t, obs['carbon'])
            if t == step:
                break
            record = json.loads(next(turns))
            assert record['public_observation_sha256'][seat] == frame['public_observation_sha256'][seat]
            actual = record['calls'][seat]
            assert actual['status'] == 'ok'
            tracker.set_actions(obs['players'], actual['response']['action'])
            prev = obs['players']
    assert prev is not None and digest(obs) == r['public_sha']
    spec = importlib.util.spec_from_file_location('terminal_teacher_close_snapshot', CLOSE/'snapshot.py')
    snapshot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snapshot)
    state_raw = (CLOSE / 'state' / f'{seed}_{seat}_{step}.json.gz').read_bytes()
    assert sha(state_raw) == SNAPSHOTS[seed]
    saved = snapshot.decode(json.loads(gzip.decompress(state_raw)), model)
    native = saved['post']['_ST']
    assert saved['post_h'] == row['snapshot_identity'] and native['prev'] == obs['players']
    assert est == native['tr'].bel.estimate() and known == list(native['tr'].bel.known)
    tracker.set_actions(obs['players'], row['orders'][0])
    assert snapshot.encode(tracker) == snapshot.encode(native['tr'])
    return obs, saved['cfg_cache'], est, prev, row['orders'], row['native'][0] - row['native'][1]
