"""Four public board views have independent neural states and one legal action prefix."""
_ONE_VIEW_PARENT = _mixed_actions


def _four_view_input(x, group, F):
    cells, moves = F.SYM_CELL[group], F.SYM_MOVE[group]
    translated = dict(x)
    translated["maps"] = x["maps"][:, cells]
    translated["tok_pos"] = cells[x["tok_pos"]]
    history = x["hist_cell"]
    translated["hist_cell"] = np.where(history >= 0, cells[np.maximum(history, 0)], history)
    translated["hist_move"] = np.append(moves, 5)[x["hist_move"]]
    return translated


def _frame_score(state, slot, unit, move, previous):
    policy, tokens, context, hidden = state
    features = np.concatenate([tokens[slot], context, unit, move.reshape(-1)])
    hidden = policy.gru(policy.lin(features, "din") + policy.w["aemb.weight"][previous], hidden)
    state[3] = hidden
    if slot < 10:
        base = policy.lin(hidden, "whead")
    else:
        base = np.concatenate([policy.lin(hidden, "bhead"), np.zeros(2, np.float32)])
    return base + policy.lin(move, "V") @ policy.lin(hidden, "U")


def _joint_frame_decode(x, aux, frames, student_weight, residual):
    F = sys.modules["cnet2_feat"]
    prefix = F.Prefix(x, aux)
    states = [(moves, [list(item) for item in nets]) for moves, nets in frames]
    previous = [8] * len(frames)
    actions = np.full(F.OWN, -1, np.int64)
    for slot in range(F.OWN):
        if not x["tok_mask"][slot]:
            continue
        unit, move, legal = prefix.feats(slot)
        totals = np.zeros((2, 5), np.float32)
        for index, (moves, nets) in enumerate(states):
            frame_move = move[moves] if slot < 10 else move
            for net_idx, state in enumerate(nets):
                logits = _frame_score(state, slot, unit, frame_move, previous[index])
                totals[net_idx] += logits[moves] if slot < 10 else logits
        expert, student = totals / len(frames)
        joint = (1.0 - student_weight) * expert + student_weight * student
        if residual and slot < 10 and x["tok_kind"][slot] == 0:
            joint = joint + _return_basis(x, aux, slot, move) @ _RETURN_THETA
        if not np.any(legal) or not np.isfinite(joint[legal]).all():
            raise ValueError("invalid shared-prefix logits or empty legal action set")
        action = int(np.argmax(np.where(legal, joint, -np.inf)))
        if not residual:
            ep = int(np.argmax(np.where(legal, expert, -np.inf)))
            sp = int(np.argmax(np.where(legal, student, -np.inf)))
            role = "base" if slot >= 10 else ("planter" if x["tok_kind"][slot] == 1 else "collector")
            counts = _COUNTS[role]
            counts["valid_slots"] += 1
            counts["disagree"] += int(ep != sp)
            counts["selected_student_in_disagreement"] += int(ep != sp and action == sp)
            counts["selected_third_in_disagreement"] += int(ep != sp and action not in (ep, sp))
        actions[slot] = action
        prefix.advance(slot, action)
        for index, (moves, _) in enumerate(states):
            previous[index] = int(moves[action]) if slot < 10 else action + 5
    return actions


def _ensemble_actions(x, aux, groups=(0, 1, 2, 3), student_weight=None):
    F = sys.modules["cnet2_feat"]
    if student_weight is None:
        student_weight = _HYBRID["student_weight"]
    frames = []
    for group in groups:
        features = _four_view_input(x, group, F) if group else x
        nets = []
        for policy in _NETS:
            tokens, context = policy.encode(features, _HYBRID["style"])
            nets.append((policy, tokens, context, policy.lin(context, "h0")))
        frames.append((F.SYM_MOVE[group], nets))
    plain = _joint_frame_decode(x, aux, frames, student_weight, False)
    if not np.any(_RETURN_THETA):
        return plain
    revised = _joint_frame_decode(x, aux, frames, student_weight, True)
    protected = (np.arange(F.OWN) >= 10) | (x["tok_kind"][:F.OWN] == 1)
    if np.any(plain[protected] != revised[protected]):
        return plain
    if np.any(plain != revised) and _return_conflicts(x, revised) - _return_conflicts(x, plain):
        return plain
    return revised


_mixed_actions = _ensemble_actions
