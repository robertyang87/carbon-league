"""Single V5 actor. Its public observations and seat history feed the production guard."""

_policy = None
_seats = {}


def agent(observation, configuration=None):
    global _policy
    if _policy is None:
        weights = _dequantize(zlib.decompress(base64.b64decode(_B1)), _Q1)
        weights["meta"] = np.array(_M1)
        _policy = _S.CachePolicy(weights)
    seat, step = observation["player"], observation["step"]
    if seat not in _seats or step <= _seats[seat][0].step:
        _seats[seat] = (_P["cnet2_feat"].Feat(seat), _S.BankruptcyGuard(CONF),
                        dict(calls=0, active_slots=0, unchanged_slots=0, changed_roots=0,
                             guard_moves=0, guard_recruit=0))
    feat, guard, counts = _seats[seat]
    x = feat.observe(observation)
    selected, logits, _ = _policy.act(x, feat.aux, teacher=CONF["TEACHER"])
    guard.reset_step()
    guarded = guard.apply(x, feat.aux, selected, logits, _policy, feat.own_uid, CONF["TEACHER"], counts)
    active = x["tok_mask"][:len(selected)]
    changed = np.count_nonzero(guarded[active] != selected[active])
    counts["calls"] += 1
    counts["active_slots"] += int(active.sum())
    counts["unchanged_slots"] += int(active.sum() - changed)
    counts["changed_roots"] += int(changed > 0)
    agent.last_stats = counts
    return feat.to_actions(guarded)
