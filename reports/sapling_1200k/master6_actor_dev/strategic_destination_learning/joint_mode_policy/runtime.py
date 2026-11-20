def _normalize_mode(value):
    delta = value - np.max(value, axis=-1, keepdims=True)
    return delta - np.log(np.exp(delta).sum(axis=-1, keepdims=True))


def _mode_prior(context):
    vector = context / np.sqrt(np.mean(context**2) + 1e-6)
    return _normalize_mode(_MODE_DATA["route.weight"] @ vector + _MODE_DATA["route.bias"])


def _mode_score(prior, base, role, move, legal):
    effect = _MODE_DATA["moves"][role] @ move.T + _MODE_DATA["bias"][role]
    logp = _normalize_mode(np.where(legal[None, :], base[None, :] + effect, -1e9))
    mixed = np.logaddexp.reduce(prior[:, None] + logp, axis=0)
    return np.where(legal, mixed, -np.inf), logp
