"""Rebuild the compact opponent-intent intervention from the frozen Sapling/it50 pack."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT/'data/weights/sapling_1200k/hybrid_it50_v3.py'
PARENT_SHA = 'b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca'


def build(out):
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('parent identity')
    src = PARENT.read_text()
    for before,after in (
        ('def _mixed_actions(x, aux):','def _raid_decode(x, aux, changes=None):'),
        ('        labels[slot] = action\n','        if changes is not None:\n            action = _raid_bonus(F, x, prefix, slot, action, combined, legal, changes)\n        labels[slot] = action\n'),
        ('def agent(observation, configuration=None):','def _raid_agent(observation, configuration=None):')
    ):
        if src.count(before) != 1:
            raise ValueError('ambiguous patch site')
        src = src.replace(before, after)
    fit = json.loads((HERE/'tree_raid_fit.json').read_text())
    src += '\n' + (HERE/'tree_raid_model.py').read_text()
    src += '\n_RAID_WEIGHTS = ' + repr(fit['weights']) + '\n'
    approach = json.loads((HERE/'tree_raid_approach.json').read_text())
    src += '\n_RAID_ENTRY_WEIGHTS = ' + repr(approach['weights']) + '\n'
    src += (HERE/'tree_raid_runtime.py').read_text()
    compile(src, str(out), 'exec')
    data = src.encode()
    if len(data) >= 10_000_000:
        raise ValueError('package exceeds limit')
    if out.exists() and out.read_bytes() != data:
        raise ValueError('different output exists')
    if not out.exists():
        out.write_bytes(data)
    print(hashlib.sha256(data).hexdigest(), len(data))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out', type=Path, required=True)
    build(cli.parse_args().out)
