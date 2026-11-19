"""Probe the original source through eval_v53's loader with and without PyTorch.

Run on a CPU with the repository dependencies, from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python reports/sapling_1200k/master3s2_dependency_probe.py
"""
import builtins
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from eval_v53 import CFG, load_agent_fn
from fastsim import FastCarbon

SOURCE = ROOT / 'data/master3s2_source.py'
SEED = 620154748
CALLS = 8


def probe(block_torch):
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name.split('.')[0] == 'torch':
            raise ModuleNotFoundError('torch intentionally unavailable for this fixture')
        return original_import(name, *args, **kwargs)

    context = (patch('builtins.__import__', import_without_torch)
               if block_torch else contextlib.nullcontext())
    with context:
        policy = load_agent_fn(str(SOURCE))
        state = policy.__globals__
        sim = FastCarbon(CFG).reset(seed=SEED)
        nonempty = 0
        for _ in range(CALLS):
            act = policy(sim.obs(0), sim.config)
            assert isinstance(act, dict)
            nonempty += bool(act)
            sim.step(act, {})
    stats = state['_S_STATS']
    assert state['_TEMP'] == 1 and state['_SEED'] is None
    assert state['_SEARCH_OK'] == (not block_torch)
    if block_torch:
        assert stats['searched'] == stats['single'] == 0
    else:
        assert stats['searched'] > 0 and stats['searched'] + stats['single'] == CALLS
    return {'torch_import_blocked': block_torch, 'search_ok': bool(state['_SEARCH_OK']),
            'calls_with_valid_dict_return': CALLS, 'nonempty_returned_actions': nonempty,
            'searched': stats['searched'], 'single': stats['single'],
            'reported_agent_last_stats': getattr(policy, 'last_stats', None)}


if __name__ == '__main__':
    for thread_var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(thread_var) != '1':
            raise SystemExit(f'Set {thread_var}=1 before launching Python.')
    digest = hashlib.md5(SOURCE.read_bytes()).hexdigest()
    assert digest == '57f9a594cce4f2ec9e3573d8d9d26ca1'
    print(json.dumps({'source_md5': digest, 'map': SEED, 'seat': 0,
                      'opponent': 'idle', 'scope': 'Local sensitivity probe; original unseeded opponent, counts may vary.',
                      'conditions': [probe(block_torch=True), probe(block_torch=False)]}, indent=2))
