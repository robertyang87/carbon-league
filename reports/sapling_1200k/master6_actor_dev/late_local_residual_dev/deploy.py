"""Appended to the frozen combined actor by build.py. All state below belongs to the residual wrapper."""

_LOCAL_PARENT_AGENT = agent
_LOCAL_PARENT_MIXED = _mixed_actions
_LOCAL_ENABLED = True
_LOCAL_PARENT_CONTEXT = None
_LOCAL_TRACE = None


def _mixed_actions(x, aux):
    global _LOCAL_PARENT_CONTEXT
    labels = _LOCAL_PARENT_MIXED(x, aux)
    _LOCAL_PARENT_CONTEXT = x, aux
    return labels


def _local_window(step, horizon):
    # The learned coefficient is supported only at actual physical turns 286..296.
    # On short games, also stop before the last two playable decisions.
    return 286 <= step <= min(296, horizon - 4)


def _local_edit(observation, cfg, parent, proposal, x, aux, features):
    if not isinstance(proposal, dict):
        return parent, "malformed_proposal"
    if not proposal:
        return parent, "no_proposal"
    if len(proposal) != 1:
        return parent, "multiple_edits"
    uid, move = next(iter(proposal.items()))
    me = observation["player"]
    mine = observation["players"][me]
    if uid not in mine[2] or mine[2][uid][2] != "COLLECTOR":
        return parent, "inactive_or_protected_target"
    if move not in ("NONE", "UP", "RIGHT", "DOWN", "LEFT"):
        return parent, "invalid_move"
    if move == parent.get(uid, "NONE"):
        return parent, "already_selected"
    alternative = dict(parent)
    if move == "NONE":
        alternative.pop(uid, None)
    else:
        alternative[uid] = move
    original = features.labels(parent)
    labels = features.labels(alternative)
    if (features.to_actions(labels) != alternative or features.to_actions(original) != parent
            or np.count_nonzero(original != labels) != 1):
        return parent, "changed_other_slot"
    f = sys.modules["cnet2_feat"]
    prefix = f.Prefix(x, aux)
    for slot in np.flatnonzero(x["tok_mask"][:f.OWN]):
        _, _, legal = prefix.feats(slot)
        order = int(labels[slot])
        if order < 0 or order >= len(legal) or not legal[order]:
            return parent, "illegal_after_prefix"
        prefix.advance(slot, order)
    if _return_conflicts(x, labels) - _return_conflicts(x, original):
        return parent, "new_friendly_or_recruit_collision"

    # The existing Prefix models Carbon League's standard 30-cost base commands.
    # Recheck visible funds and capacity in the game's actual sequential base order.
    members = [w for team in observation["players"] for w in team[2].values()]
    ncol = sum(w[2] == "COLLECTOR" for w in members)
    prices = {"RECCOLLECTOR": cfg["recCollectorCost"] + cfg["recCollectorIncreaseCost"] * ncol,
              "RECPLANTER": cfg["recPlanterCost"] + cfg["recPlanterIncreaseCost"] * (len(members) - ncol)}
    cash = mine[0]
    count = len(mine[2])
    for base in mine[1]:
        command = alternative.get(base)
        if command is None:
            continue
        if command not in prices or count >= cfg["workerLimit"] or cash < prices[command]:
            return parent, "unfunded_or_over_capacity_recruit"
        cash -= prices[command]
        count += 1
    return alternative, "accepted"


def agent(observation, configuration=None):
    global _LOCAL_TRACE, _LOCAL_PARENT_CONTEXT
    _LOCAL_PARENT_CONTEXT = None
    parent = _LOCAL_PARENT_AGENT(observation, configuration)
    step = observation["step"]
    cfg = {k:(configuration or {}).get(k,v) for k,v in _LOCAL_DEFAULT_CFG.items()}
    proposal = {}
    action = parent
    if not _LOCAL_ENABLED:
        reason = "disabled"
    elif not _local_window(step, cfg["episodeSteps"]):
        reason = "outside_window"
    else:
        # This object is deliberately limited to the platform's acting-seat information.
        view = {k:observation[k] for k in ("step","player","players","trees","carbon")}
        proposal = _LOCAL_RUNTIME.residual_action(view, cfg, dict(parent), _LOCAL_WEIGHTS)
        x, aux = _LOCAL_PARENT_CONTEXT
        action, reason = _local_edit(view, cfg, parent, proposal, x, aux, _FEAT)
    _LOCAL_TRACE = {"step":step,"seat":observation["player"],"parent":parent,"proposal":proposal,"reason":reason}
    _LOCAL_PARENT_CONTEXT = None
    return action
