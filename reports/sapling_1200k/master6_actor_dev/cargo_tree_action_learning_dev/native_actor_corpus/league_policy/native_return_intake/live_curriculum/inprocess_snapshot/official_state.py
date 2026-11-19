"""In-process checkpoints of the official Carbon referee, request budget and process RNGs.

The caller owns exclusive use of this interpreter and the process RNGs. Player modules'
private generators and the examiner's real hardware clock are outside this referee state.
"""
import copy
import hashlib
import pickle
import random

import numpy as np


_STATIC = frozenset(("interpreter", "renderer", "html_renderer", "agents", "pool"))


def _ids():
    from zerosum_env.envs.carbon import idgen
    return idgen


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        return None
    return torch


def _state(env, bank):
    if env.pool is not None:
        raise ValueError("A live agent pool cannot share a process-local referee checkpoint")
    if not env.steps or env.states is not env.steps[-1]:
        raise ValueError("Official state is detached from the most recent recorded step")
    torch = _torch()
    with _ids()._rlock:
        ids = dict(_ids()._id_dict)
    # A single deepcopy memo preserves state/steps and all nested Struct attribute aliases.
    mutable = {k: v for k, v in vars(env).items() if k not in _STATIC}
    return dict(environment=mutable, bank=bank, ids=ids,
                default_step_log=type(env).step.__defaults__[0], python_rng=random.getstate(),
                numpy_rng=np.random.get_state(), torch_rng=None if torch is None else
                torch.get_rng_state().numpy().tobytes())


def take(env, bank):
    """Capture history, observations, errors, configuration, counters and both time banks."""
    return env, env.logs, copy.deepcopy(_state(env, bank))


def restore(env, state):
    """Restore the same referee in place and return an independent copy of the saved bank."""
    owner, logbook, saved = state
    if owner is not env or env.pool is not None:
        raise ValueError("Checkpoint belongs to a different official environment or process pool")
    # Environment.step(actions, logs=[]) appends its own persistent default list.
    # Preserve both external list identities and their aliases to historical entries.
    template = saved["default_step_log"]
    default_log = type(env).step.__defaults__[0]
    anchors = {id(template): (template, default_log),
               id(saved["environment"]["logs"]): (saved["environment"]["logs"], logbook)}
    memo = {key: target for key, (_, target) in anchors.items()}
    for source, target in anchors.values():
        target[:] = copy.deepcopy(list(source), memo)
    current = copy.deepcopy(saved, memo)
    for k in vars(env).keys() - _STATIC - current["environment"].keys():
        delattr(env, k)
    vars(env).update(current["environment"])
    ids = _ids()
    with ids._rlock:
        ids._id_dict.clear()
        ids._id_dict.update(current["ids"])
    torch = _torch()
    if (torch is None) != (current["torch_rng"] is None):
        raise ValueError("Torch CPU RNG availability changed during an in-process branch")
    if torch is not None:
        rng = np.frombuffer(current["torch_rng"], dtype=np.uint8).copy()
        torch.set_rng_state(torch.from_numpy(rng))
    random.setstate(current["python_rng"])
    np.random.set_state(current["numpy_rng"])
    if env.states is not env.steps[-1]:
        raise ValueError("Restored referee history has lost its current-state alias")
    return current["bank"]


def fingerprint(env, bank):
    """Same-process audit digest, including full history and the saved global RNG streams."""
    value = _state(env,bank)
    # Serialize the same independent graph as take(), including object references, dictionary
    # order and the official Struct's separate attribute and item tables.
    return hashlib.sha256(pickle.dumps(copy.deepcopy(value),protocol=5)).hexdigest()
