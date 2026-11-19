"""Shared, torch-free helpers for the KataGo warm-start tooling (plan stages 0-3).

This module is the single source of truth for the master3s2 weight layout, so
that extract_m3s2.py, model.py, train.py, distill_roll.py and pack_submission.py
cannot drift apart.  Everything here is grounded in the actual source files:

  * The chassis/policy payload `_WEIGHTS` is an npz (zip of .npy, deflate
    compressed -- i.e. what numpy.savez_compressed writes) carried as one
    base64 string.  Arrays are stored as float16; `_Policy.__init__` casts
    them to float32 on load.  Keys: 'snames' (120 strategy names, <U33),
    'meta' (int32, meta[0] = number of residual blocks, 6), 'stem.w/b',
    'emb' (len(snames)+1 rows; sid() may return len(snames) for unknown
    names), per block i: 'b{i}.c1.w/b', 'b{i}.g.w/b', 'b{i}.c2.w/b',
    'hw.w' (5, C, 1, 1), 'hw.b', 'hb.w' (3, C, 1, 1), 'hb.b'.
  * The critic payload `_VALUE` holds 'v1.weight' (K, C, 1, 1), 'v1.bias',
    'v2.weight' (H, 2K+5), 'v2.bias', 'v3.weight' (1, H), 'v3.bias',
    with K=32, H=128 in the shipped weights.
  * NOTE: the in-file `_quant(x) = clip(rint(x*100), 0, 255) * 0.01` is the
    *feature* quantizer applied to encode() planes in _act/_search.  The
    source does NOT int8-quantize weights; weights are stored float16.
    pack_submission.py therefore stores float16, exactly like the source.

Also provides the game-driver utilities shared by the pack verifier and
vec_selfplay: new_game_state() replicates the environment's populate_board
(zerosum_env/envs/carbon/carbon.py) using only DEFAULT_CONFIG fields of the
chassis module, so self-games start from the same distribution the ladder
uses (V4-symmetric carbon quartiles, bases at the four startPosOffset
positions, starting cash = the chassis' START_CASH, 80).
"""

import base64
import io
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(HERE, '..', '..', 'src'))
MASTER3S2 = os.path.join(SRC_DIR, 'master3s2.py')
FIX2 = os.path.join(SRC_DIR, 'apexcarbon_v1_fix2.py')
APEX_V4TTA = os.path.join(SRC_DIR, 'apexcarbon_v4tta_search.py')

VALUE_KEYS = ('v1.weight', 'v1.bias', 'v2.weight', 'v2.bias', 'v3.weight', 'v3.bias')
BLOCK_KEYS = ('c1.w', 'c1.b', 'g.w', 'g.b', 'c2.w', 'c2.b')
SID_NAME = 'TJB-o18-rush-v41'          # the strategy row the chassis plays (_SID)


def policy_keys(nb):
    """All _WEIGHTS array keys (besides 'snames'/'meta') for nb blocks, in source order."""
    ks = ['stem.w', 'stem.b', 'emb']
    for i in range(nb):
        ks += ['b%d.%s' % (i, k) for k in BLOCK_KEYS]
    ks += ['hw.w', 'hw.b', 'hb.w', 'hb.b']
    return ks


def load_payload_b64(source_text, name):
    """Extract one base64 payload string ('_WEIGHTS'/'_VALUE'/'_ROLL') from module source."""
    m = re.search(r'^%s = "([A-Za-z0-9+/=]+)"' % re.escape(name), source_text, re.M)
    return m.group(1) if m else None


def decode_npz(b64):
    """base64 payload -> {key: ndarray} (dtypes as stored)."""
    z = np.load(io.BytesIO(base64.b64decode(b64)), allow_pickle=False)
    return {k: z[k] for k in z.files}


def encode_npz(arrays):
    """{key: ndarray} -> base64 payload string, in the source's own format
    (numpy.savez_compressed: a deflate zip of .npy members)."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def load_source_payloads(path):
    """Read a chassis/policy module file -> (policy, value, snames, nb).

    policy/value are {key: float32 ndarray} exactly as _Policy/_TorchNet see
    them (the stored float16 cast to float32); snames is a list of str;
    nb = int(meta[0]).  value is None when the file has no _VALUE payload.
    """
    text = open(path).read()
    pol = decode_npz(load_payload_b64(text, '_WEIGHTS'))
    snames = [str(s) for s in pol.pop('snames')]
    nb = int(pol.pop('meta')[0])
    policy = {k: v.astype(np.float32) for k, v in pol.items()}
    vb = load_payload_b64(text, '_VALUE')
    value = None
    if vb is not None:
        vz = decode_npz(vb)
        value = {k: v.astype(np.float32) for k, v in vz.items()}
    return policy, value, snames, nb


def split_checkpoint(arrays):
    """Split a full checkpoint dict (source layout, incl. 'snames'/'meta')
    into (policy_arrays, value_arrays) ready for encode_npz."""
    pol = {k: v for k, v in arrays.items() if k not in VALUE_KEYS}
    val = {k: v for k, v in arrays.items() if k in VALUE_KEYS}
    return pol, val


def to_f16(arrays):
    """Cast float arrays to float16 (the storage dtype); leave snames/meta alone."""
    out = {}
    for k, v in arrays.items():
        if k in ('snames', 'meta'):
            out[k] = v
        else:
            out[k] = np.ascontiguousarray(v, dtype=np.float16)
    return out


def inventory_lines(policy, value, snames, nb):
    """Human-readable key/shape/dtype inventory of a payload pair."""
    lines = ['snames         <U{}       ({},)'.format(
        max((len(s) for s in snames), default=0), len(snames)),
        'meta           int32      (1,)  value=[%d]  (nb blocks)' % nb]
    for k in policy_keys(nb):
        a = policy[k]
        lines.append('%-14s %-10s %s' % (k, a.dtype, a.shape))
    if value is not None:
        for k in VALUE_KEYS:
            a = value[k]
            lines.append('%-14s %-10s %s' % (k, a.dtype, a.shape))
    return lines


def load_module(path, name):
    """Import a python file as a module under `name` (used for chassis files)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def new_game_state(mod, seed):
    """A fresh game State, replicating zerosum_env's populate_board().

    mod: an imported chassis module (provides State, DEFAULT_CONFIG and,
    in master3s2, START_CASH).  Carbon is dealt into quartiles with the
    env's exact calls (np.random.uniform(2*mean) / choice over the half
    grid, int truncation, four-way mirroring); base cells are zeroed;
    player p's bases sit at starting_positions[2p], [2p+1]; both players
    start with START_CASH cash and no workers or trees.
    """
    cfg = mod.DEFAULT_CONFIG
    size = cfg['size']
    half = (size + 1) // 2                     # math.ceil(size / 2)
    np.random.seed(seed)
    min_cells = int(np.ceil(cfg['startingCarbon'] / cfg['startingCellCarbon'] / 4))
    n_cells = int(np.ceil(half * half * cfg['carbonCoverage'] / 100))
    n_cells = max(n_cells, min_cells)
    cmean = min(cfg['startingCarbon'] / 4 / n_cells, cfg['startingCellCarbon'])
    vals = np.random.uniform(2 * cmean, size=n_cells)
    idx = np.random.choice(half * half, size=n_cells, replace=False)
    hg = np.zeros(half * half, np.float16)
    hg[idx] = vals
    hg = hg.reshape((half, half))
    carbon = [0] * (size * size)
    for r, row in enumerate(hg):
        for c, v in enumerate(row):
            v = min(int(v), cfg['startingCellCarbon'])
            carbon[size * r + c] = v
            carbon[size * r + (size - c - 1)] = v
            carbon[size * (size - 1) - (size * r) + c] = v
            carbon[size * (size - 1) - (size * r) + (size - c - 1)] = v
    off = cfg['startPosOffset']
    pos = [size * (size // 4 + off) + size // 4 + off,
           size * (3 * size // 4 - off) + 3 * size // 4 - off,
           size * (size // 4 + off) + 3 * size // 4 - off,
           size * (3 * size // 4 - off) + size // 4 + off]
    for p in pos:
        carbon[p] = 0
    cash = float(getattr(mod, 'START_CASH', 80))
    players = []
    for i in range(2):
        bases = {'player-%d-base-%d' % (i, j): pos[i * cfg['numberOfBases'] + j]
                 for j in range(cfg['numberOfBases'])}
        players.append([cash, bases, {}, {}])
    return mod.State(0, [float(c) for c in carbon], players, {}, {})


def make_obs(state, me):
    """An observation dict in the shape the chassis agent() consumes.

    The harness is full-information: obs['carbon'] is the true map (the
    chassis' BeliefTracker then learns exact values at visible cells).
    """
    return {'step': state.step, 'player': me, 'players': state.players,
            'carbon': list(state.carbon), 'trees': state.trees_meta,
            'remainingOverageTime': 0}
