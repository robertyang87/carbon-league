"""Fail-closed numerical checks and durable PPO checkpoint publication."""
import hashlib
import json
import math
import os
import tempfile

import numpy as np


def require_finite(value, label):
    """Check complete model/optimizer trees, naming the first bad tensor."""
    import torch
    groups = {}
    def visit(item, name):
        if isinstance(item, torch.Tensor):
            groups.setdefault(item.device, []).append((name, torch.isfinite(item).all()))
        elif isinstance(item, np.ndarray):
            if not np.isfinite(item).all():
                raise FloatingPointError(f"non-finite {name}")
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{name}.{key}")
        elif isinstance(item, (tuple, list)):
            for key, child in enumerate(item):
                visit(child, f"{name}[{key}]")
        elif isinstance(item, (float, np.floating)) and not math.isfinite(item):
            raise FloatingPointError(f"non-finite {name}")
    visit(value, label)
    # One host synchronization per device on the healthy path, rather than
    # hundreds of per-parameter CUDA synchronizations in every minibatch.
    for entries in groups.values():
        flags = torch.stack([flag for _, flag in entries])
        if not flags.all().item():
            bad = (~flags).nonzero()[0].item()
            raise FloatingPointError(f"non-finite {entries[bad][0]}")


def _sync_dir(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(value, path):
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix='.json-', dir=directory)
    try:
        with os.fdopen(fd, 'w') as fh:
            json.dump(value, fh, allow_nan=False)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
        _sync_dir(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_torch_save(value, path, previous_path=None):
    """Never truncate the last recovery point, even if serialization fails.

    Keep the previous inode as a second recovery point before replacing the
    current one. All checkpoint writers use replace, so the hard link cannot
    be mutated by the next save. The temporary file is on the same filesystem.
    """
    import torch
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix='.state-', dir=directory)
    backup = None
    try:
        with os.fdopen(fd, 'wb') as fh:
            torch.save(value, fh)
            fh.flush(); os.fsync(fh.fileno())
        if previous_path and os.path.exists(path):
            backup = tmp + '.previous'
            os.link(path, backup)
            os.replace(backup, previous_path)
        os.replace(tmp, path)
        _sync_dir(directory)
    finally:
        for leftover in (tmp, backup):
            if leftover and os.path.exists(leftover):
                os.unlink(leftover)


def commit_checkpoint(out, state, actor_path, phase):
    """The manifest is the completion marker; train rows come AFTER this.

    The full state is authoritative for actor, critic and optimizers. A crash
    between replacing state/actor and publishing the marker leaves no matching
    completion marker; consumers can verify the hashes rather than guessing.
    """
    require_finite(state, 'checkpoint')
    with open(actor_path, 'rb') as actor:
        os.fsync(actor.fileno())
    path = os.path.join(out, 'state_last.pt')
    atomic_torch_save(state, path, os.path.join(out, 'state_previous.pt'))
    marker = {'it': state['it'], 'phase': phase,
              'files': {os.path.basename(p): sha256_file(p) for p in (path, actor_path)}}
    atomic_json(marker, os.path.join(out, 'checkpoint_complete.json'))
    return marker
