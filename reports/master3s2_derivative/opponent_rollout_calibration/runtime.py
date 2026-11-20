"""Modify only the opponent's simulated policy pre-decode logits at active locations."""
def adjust(players, op, worker_logits, base_logits, beta):
    # Caller may share a batch with other successors, so keep arrays read-only.
    worker = worker_logits.copy()
    base = base_logits.copy()
    for pos, cargo, role in players[op][2].values():
        worker[pos] *= beta[role]
    for pos in players[op][1].values():
        base[pos] *= beta["BASE"]
    return worker, base
