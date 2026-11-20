"""Public observer encodings for four archived responses from a single exposed map."""
import copy
import gzip
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DERIVATIVE = HERE.parent
ROOT = DERIVATIVE.parents[1]
ARCHIVE = DERIVATIVE / 'ladder/v4_alphacarbon_live_response'
LADDER_FILE = DERIVATIVE / 'ladder/v4_observations/increment_0904/replays/6e994d80674dc2430a4925b71bb6a947-1.json.gz'
P_HASH = 'af4938882c206c56d977417a76e2b54051e1aecfe7d1092d146c4cda90c4b2a1'


def signed(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'Different frozen bytes: {path}')
    return raw


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cards(steps=(268, 269, 270)):
    """Return pre-action enemy X, public players and a separate observed action label.

    An official referee in this offline reader projects the public requests from
    already stored actions. Only V4's seat-zero projected carbon, shared assets,
    and *previous* V4 orders enter its belief. The enemy label in this output is
    a newly drawn decision of the registered repository original on that suffix.
    All four suffixes share the same original ladder map and opening position.
    """
    if set(steps) - {268, 269, 270}:
        raise ValueError('Exposure limited to the three registered interpretation steps')
    if not steps:
        return []
    protocol = json.loads(signed(ARCHIVE / 'protocol.json', P_HASH))
    results = json.loads(signed(ARCHIVE / 'results.json', '30b38461adefd576fee0562dbba05dcce39c11c4d30cabe64a2ec59fd95b0977'))
    audit = json.loads(signed(ARCHIVE / 'audit.json', 'f52d2fa1fe83017f67ccb437956ca6833c3cd42d24be4ace4a6fcf6dc82e7e20'))
    if results['protocol_sha256'] != P_HASH or audit['protocol_sha256'] != P_HASH:
        raise ValueError('Different suffix protocol')
    parent = protocol['executables']['v4']
    signed(ROOT / parent['path'], parent['sha256'])
    tape = json.loads(gzip.decompress(signed(LADDER_FILE, '91fc071bd9e395d02297c73f03835711832cec8b41b73ad562aa54268e3e3fbb')))
    sys.path.insert(0, str(ROOT / 'code/env_src/carbon_challenge-master'))
    family = module(DERIVATIVE / 'freight_response/run.py', 'evasion_alpha_public')
    net = module(ROOT / parent['path'], 'evasion_alpha_v4')
    net._net()
    stream_rng = np.random.get_state(), random.getstate()
    try:
        seed = tape['configuration']['randomSeed']
        random.seed(seed)
        np.random.seed(seed)
        env = family.make('carbon', configuration=tape['configuration'])
        frames = tape['steps']
        # Initial hidden map is referee-only. It cannot be read by build_features.
        env.states[0].observation.full_carbon = copy.deepcopy(frames[0][0]['observation']['full_carbon'])
        tr = net.BeliefTracker(net._Bel, 0)
        prev = None
        for t in range(protocol['step']):
            pub = frames[t][0]['observation']
            orders = [p['action'] or {} for p in frames[t+1]]
            tr.step(pub['players'], t, pub['carbon'])
            tr.set_actions(pub['players'], orders[0])
            prev = pub['players']
            env.step(orders, [])
        root = protocol['step']
        env.steps[:-1] = [None]*root
        env.logs = []
        state = family.official_state.take(env, [s['observation']['remainingOverageTime'] for s in frames[root]])
        output = []
        for trial in results['results']:
            record = json.loads(gzip.decompress(signed(ARCHIVE / trial['artifact'], trial['sha256'])))
            if record['protocol_sha256'] != P_HASH:
                raise ValueError('Different suffix identity')
            previous = copy.deepcopy(prev)
            tracker = copy.deepcopy(tr)
            bank = family.official_state.restore(env, state)
            for i, r in enumerate(record['rows']):
                t = root+i
                if t > max(steps):
                    break
                o = family.public(env, 0, bank)
                if t != r['step'] or family.digest(o) != r['public_sha256'][0]:
                    raise ValueError(f'Observer state mismatch at {t}')
                players = o['players']
                est, _ = tracker.step(players, t, o['carbon'])
                if t in steps:
                    # Matches V4 _search(xo): the rival at root is encoded using
                    # OUR estimate and the original all-ones second input plane.
                    x = net._quant(net.encode(players, t, 1, est, [1.]*net.N, previous)).astype(np.float32)
                    who = protocol['worker_by_seat'][1]
                    if r['orders'][1] != r['native_orders'][1] or who not in players[1][2]:
                        raise ValueError('Missing natural foe order or carrier')
                    output.append(dict(stream=trial['stream'], arm=trial['arm'], terrain=protocol['map'],
                                       observer=0, enemy=1, seat=0, seed=seed, step=t, worker=who,
                                       x=x, players=copy.deepcopy(players), prev_players=copy.deepcopy(previous),
                                       public_sha256=r['public_sha256'][0],
                                       label=r['orders'][1].get(who,'NONE'),
                                       enemy_order=copy.deepcopy(r['orders'][1]), source_sha256=trial['sha256']))
                # The current two orders affect only the *next* state and the
                # current action label above. No policy search is called here.
                tracker.set_actions(players, r['orders'][0])
                previous = copy.deepcopy(players)
                env.step(r['orders'], [])
                bank = [b-max(0,c['wall']-1) for b,c in zip(bank, r['clocks'])]
        if len(output) != len(set(steps))*4:
            raise ValueError('Incomplete archived stream coverage')
        return output
    finally:
        np.random.set_state(stream_rng[0])
        random.setstate(stream_rng[1])
