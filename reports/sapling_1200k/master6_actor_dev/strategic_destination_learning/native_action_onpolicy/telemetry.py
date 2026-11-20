# Training observations only. The public command is sampled by the original frozen menu.
_onpolicy_distribution = _native_distribution
_onpolicy_mixed = _mixed_actions
_ONPOLICY_FEATURES = None


def _native_distribution(base, menu):
    global _ONPOLICY_FEATURES
    teams, probabilities = _onpolicy_distribution(base, menu)
    _ONPOLICY_FEATURES = None
    if menu is not None:
        pivot, actions, full_teams, features, logits = menu
        keys = {tuple(sorted(_FEAT.to_actions(team).items())): i for i, team in enumerate(teams)}
        h = np.tanh(features[actions] @ _NATIVE_W0 + _NATIVE_B0)
        _ONPOLICY_FEATURES = {
            "hidden": h.tolist(), "logits": logits[actions].tolist(), "actions": actions.tolist(),
            "base_index": keys[tuple(sorted(_FEAT.to_actions(base).items()))],
            "route_to_order": [keys[tuple(sorted(_FEAT.to_actions(team).items()))] for team in full_teams],
        }
    return teams, probabilities


def _mixed_actions(x, aux):
    team = _onpolicy_mixed(x, aux)
    _LAST_POLICY["policy_features"] = _ONPOLICY_FEATURES
    return team
