# Include the matched untrained sampler probabilities without decoding a second time.
_native_fitted_distribution = _native_distribution
_native_recorded_mixed = _mixed_actions
_NATIVE_PRIOR = {}


def _native_distribution(base, menu):
    global _NATIVE_PRIOR
    teams, probabilities = _native_fitted_distribution(base, menu)
    root = tuple(sorted(_FEAT.to_actions(base).items()))
    if menu is None:
        _NATIVE_PRIOR = {root: 1.0}
    else:
        _, actions, alternatives, _, logits = menu
        _NATIVE_PRIOR = {root: 0.5}
        for other, mass in zip(alternatives, _native_softmax(logits[actions])):
            key = tuple(sorted(_FEAT.to_actions(other).items()))
            _NATIVE_PRIOR[key] = _NATIVE_PRIOR.get(key, 0.) + .5 * float(mass)
    return teams, probabilities


def _mixed_actions(x, aux):
    labels = _native_recorded_mixed(x, aux)
    for item in _LAST_POLICY["orders"]:
        item["prior_probability"] = _NATIVE_PRIOR[tuple(sorted(item["action"].items()))]
    return labels
