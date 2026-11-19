"""Appended only to the training build. The exported greedy pack has none of these hooks."""
import hashlib
import json
import os
from pathlib import Path

_SAMPLE_STEPS = (12, 48, 84, 120, 156, 192, 228, 264)
_original_mixed = _mixed_actions
_normal_agent = agent
_capture = None
_forced = None
_obs_key = None
last_training_event = None


def _sample_slot(branch, slot, predicted, hidden, scores, legal):
    if _capture is not None:
        _capture[bool(branch)][slot] = (hidden.copy(), scores.copy(), legal.copy())
    if _forced is not None and _forced[0] == slot:
        return _forced[1] if legal[_forced[1]] else predicted
    return predicted


def _physical(labels):
    return dict(_FEAT.to_actions(labels))


def _sample_branch(x, aux, action=None):
    global _capture, _forced
    captured = {False: {}, True: {}}
    _capture, _forced = captured, action
    try:
        labels = _original_mixed(x, aux)
    finally:
        _capture, _forced = None, None
    # The exact parent safety gate explicitly records its chosen decoder.
    return labels, captured[_chosen_route], _chosen_route


def _mixed_actions(x, aux):
    global last_training_event
    t, seat = _obs_key
    if t not in _SAMPLE_STEPS:
        return _original_mixed(x, aux)
    baseline, trace, route = _sample_branch(x, aux)
    menus = []
    for slot, (hidden, scores, legal) in trace.items():
        if slot >= 10 or np.count_nonzero(legal) < 2:
            continue
        options = sorted(np.flatnonzero(legal), key=lambda a: (-float(scores[a]), int(a)))
        best, second = map(int, options[:2])
        if int(baseline[slot]) != best:
            raise ValueError("recorded route does not explain the physical command")
        gap = float(scores[best] - scores[second])
        if gap <= 1.0:
            menus.append((gap, slot, best, second))
    for gap, slot, best, second in sorted(menus):
        alternate, alt_trace, alt_route = _sample_branch(x, aux, (slot, second))
        if (alt_route != route or int(alternate[slot]) != second or
                not np.array_equal(baseline[:slot], alternate[:slot]) or
                not np.array_equal(trace[slot][0], alt_trace[slot][0]) or
                not np.array_equal(trace[slot][1], alt_trace[slot][1]) or
                _return_conflicts(x, alternate) - _return_conflicts(x, baseline)):
            continue
        base_cmd, alt_cmd = _physical(baseline), _physical(alternate)
        if base_cmd == alt_cmd:
            continue
        logits = trace[slot][1][[best, second]].astype(np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        stream = int(os.environ["SAPLING_NATIVE_STREAM"])
        # An independent per-ticket stream affects only training sampling.
        uniform = float(np.random.default_rng(np.random.SeedSequence([9173951, stream, seat, t])).random())
        choice = int(uniform >= probs[0])
        issued = (base_cmd, alt_cmd)[choice]
        last_training_event = dict(step=t, seat=seat, supported=True, slot=slot,
                uid=str(_FEAT.own_uid[slot]), role=int(x["tok_kind"][slot]),
                prefix=baseline[:slot].tolist(), route=bool(route), actions=[best,second],
                legal=trace[slot][2].tolist(), logits=logits.tolist(), probabilities=probs.tolist(),
                logp=float(np.log(probs[choice])), uniform=uniform, choice=choice,
                hidden=trace[slot][0].tolist(), options=[base_cmd,alt_cmd], issued=issued)
        return (baseline, alternate)[choice]
    last_training_event = dict(step=t,seat=seat,supported=False,issued=_physical(baseline))
    return baseline


def agent(observation, configuration=None):
    global _obs_key, last_training_event
    _obs_key = int(observation["step"]), int(observation["player"])
    last_training_event = None
    command = _normal_agent(observation, configuration)
    if _obs_key[0] in _SAMPLE_STEPS:
        row = last_training_event
        if row is None or row["issued"] != command:
            raise ValueError("training event and public command differ")
        folder = os.environ.get("SAPLING_NATIVE_TRACE")
        if folder:
            path = Path(folder) / f"{_obs_key[0]:03d}.json"
            payload = json.dumps(row, sort_keys=True, separators=(",",":"), allow_nan=False).encode()
            if path.exists():
                if path.read_bytes() != payload:
                    raise ValueError("training sampling drift on resume")
            else:
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(payload)
                os.replace(tmp, path)
    return command
