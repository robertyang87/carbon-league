"""Rebuild the standalone endgame candidate from the signed v1 and two pinned sources."""
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = ROOT / 'data/weights/sapling_1200k/terminal_return_fit_v1.py'
SIM = ROOT / 'code/fastsim.py'
PIN = {PARENT: '9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0',
       SIM: '1f59c01e81dbd9f24747b5ee5770a17a581633790647d131f128b0125284f9ac',
       HERE/'policy.py': '1f056cf592db539d1fef10112f8400d66506f38049bffa09007752c5c097cc62',
       HERE/'protocol.json': '575159ddc7aceb0113ec69518cbf172cde7f2dba3a9a8abac5b2e81c284c9f44'}

RUNTIME = '''
import builtins as _queue_builtins

_QUEUE_SIM = types.ModuleType('_sapling_queue_sim')
exec(zlib.decompress(base64.b85decode(_QUEUE_SIM_SOURCE)), _QUEUE_SIM.__dict__)
_QUEUE_POLICY = types.ModuleType('_sapling_queue_policy')

def _queue_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'fastsim' and level == 0:
        return _QUEUE_SIM
    return _queue_builtins.__import__(name, globals, locals, fromlist, level)

_QUEUE_POLICY.__dict__['__builtins__'] = dict(vars(_queue_builtins), __import__=_queue_import)
exec(zlib.decompress(base64.b85decode(_QUEUE_POLICY_SOURCE)), _QUEUE_POLICY.__dict__)
_QUEUE_BASE = agent
_QUEUE_PREVIOUS = None
_QUEUE_TRACE = None

def _queue_remember(observation):
    global _QUEUE_PREVIOUS
    # Only previous public worker rosters are needed to infer each heading.
    _QUEUE_PREVIOUS = {'step': observation['step'], 'player': observation['player'],
                       'players': [(None, None, {u: tuple(w) for u, w in pl[2].items()})
                                   for pl in observation['players']]}

def agent(observation, configuration=None):
    global _QUEUE_TRACE
    base = _QUEUE_BASE(observation, configuration)
    t, seat = observation['step'], observation['player']
    cfg = _QUEUE_SIM.DEFAULT_CONFIG
    if configuration is not None:
        cfg = {k: configuration.get(k, default) for k, default in cfg.items()}
    selected, audit = base, None
    if cfg['episodeSteps'] - 1 - t in (1, 2):
        prior = _QUEUE_PREVIOUS
        past = {t-1: prior} if prior and prior['step'] == t-1 and prior['player'] == seat else {}
        selected, audit = _QUEUE_POLICY.choose(observation, past, base, cfg)
    _QUEUE_TRACE = (t, seat, dict(base), audit)
    _queue_remember(observation)
    return dict(selected)
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def frozen(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        output.write(data)
        tmp = output.name
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f'different existing build at {path}')
    finally:
        os.unlink(tmp)


def build(destination=HERE):
    destination = Path(destination)
    source = {p: p.read_bytes() for p in PIN}
    if any(sha(data) != PIN[p] for p, data in source.items()):
        raise ValueError('package input differs from signed source')
    appendix = []
    for p, symbol in ((SIM, '_QUEUE_SIM_SOURCE'), (HERE/'policy.py', '_QUEUE_POLICY_SOURCE')):
        blob = base64.b85encode(zlib.compress(source[p], level=9))
        appendix.append(f'{symbol} = {blob!r}')
    payload = source[PARENT] + ('\n'+'\n'.join(appendix)+'\n'+RUNTIME).encode()
    compile(payload, 'candidate.py', 'exec')
    limit = 10_000_000
    if len(payload) > limit:
        raise ValueError('single-file byte limit exceeded')
    manifest = {'candidate_sha256': sha(payload), 'parent_sha256': PIN[PARENT], 'bytes': len(payload),
                'limit_bytes': limit, 'builder_sha256': sha(Path(__file__).read_bytes()),
                'source_sha256': {str(p.relative_to(ROOT)): ident for p, ident in PIN.items()},
                'dependencies': ['Python standard library', 'NumPy'], 'entry_point': 'agent(observation, configuration)',
                'history': 'One previous public worker roster per side; validate adjacent step and seat.',
                'selection': 'Run v1 every call; choose from the observed live state only when one or two moves remain.'}
    frozen(destination/'candidate.py', payload)
    frozen(destination/'package.json', (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode())
    return manifest


if __name__ == '__main__':
    print(json.dumps(build(), sort_keys=True))
