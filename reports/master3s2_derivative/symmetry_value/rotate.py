"""Half-turn on a 15x15 Carbon board. Seats and entity identities stay fixed."""
import numpy as np

SIZE = 15
CELLS = SIZE * SIZE
WORKER_INVERSE = (0, 3, 4, 1, 2)
ACTION = {
    "UP": "DOWN", "RIGHT": "LEFT", "DOWN": "UP", "LEFT": "RIGHT",
    "NONE": "NONE", "RECCOLLECTOR": "RECCOLLECTOR", "RECPLANTER": "RECPLANTER", None: None,
}


def rotate_cell(pos):
    if not 0 <= pos < CELLS:
        raise ValueError("Outside the 15x15 board")
    return CELLS - 1 - pos


def rotate_players(players):
    return [[cash, {uid: rotate_cell(pos) for uid, pos in bases.items()},
             {uid: [rotate_cell(pos), cargo, role] for uid, (pos, cargo, role) in workers.items()},
             {uid: [rotate_cell(pos), age] for uid, (pos, age) in trees.items()}]
            for cash, bases, workers, trees in players]


def rotate_orders(orders):
    return {uid: ACTION[action] for uid, action in orders.items()}


def rotate_observation(obs):
    # The opaque simulation cache _st embeds the original board: never copy it through.
    # Only simulation-observation fields enter this transform. Do not propagate a referee-only field.
    return dict(step=obs['step'], player=obs['player'], players=rotate_players(obs['players']),
                carbon=list(reversed(obs['carbon'])), remainingOverageTime=obs['remainingOverageTime'],
                trees={uid: list(meta) for uid, meta in obs['trees'].items()})


def encode_rotated(model, players, step, me, carbon, known, previous=None):
    return model.encode(rotate_players(players), step, me, list(reversed(carbon)), list(reversed(known)),
                        rotate_players(previous) if previous is not None else None)


def rotate_features(x):
    """Exact spatial and directional permutation for the original 54-plane encode."""
    if x.shape[-3:] != (54, 15, 15):
        raise ValueError("Expected 54-plane, 15x15 native features")
    y = np.array(x[..., ::-1, ::-1], copy=True, order="C")
    for first in (25, 30):
        y[..., first:first+5, :, :] = y[..., [first, first+3, first+4, first+1, first+2], :, :]
    return y
