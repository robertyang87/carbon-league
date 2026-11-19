"""Action-conditional public features and safety invariants; the only fitted parameters are four coefficients."""
def _return_basis(x, aux, slot, move):
    pos = int(x["tok_pos"][slot])
    load = float(x["tok_cargo"][slot])
    loaded = load / (load + 60.0)
    d0 = float(aux["dob"][pos])
    d = np.rint(move[:, 10] * 14)
    near_end = np.clip((d0 + 12 - x["left"]) / 12, 0, 1)
    # All five actions compete at the current slot with its actual preceding decisions.
    bank = loaded * (move[:, 5] + (.5 + .5 * near_end) * np.clip(d0 - d, -1, 1))
    risk = loaded * (move[:, 1] + .25 * move[:, 4] + .25 * move[:, 13])
    # mf[2] is log1p(largest heavier rival that can immediately contest this cell)/6.5.
    prey = np.expm1(move[:, 2] * 6.5)
    attainable = (prey / (prey + 80.0)) * (1 - loaded) * (1 - move[:, 1])
    time_to_bank = np.clip((x["left"] - 1 - d) / 4, 0, 1)
    pursuit = attainable * time_to_bank * (1 - .5 * move[:, 13]) * (1 - np.clip(move[:, 0], 0, 1))
    harvest = (move[:, 11] * move[:, 7] * (1 - loaded) * (1 - move[:, 1]) *
               (1 - move[:, 5]) * (1 - move[:, 6]) * (1 - move[:, 8]) * (1 - move[:, 9]) *
               (1 - .5 * move[:, 13]) * time_to_bank)
    return np.stack((bank, risk, pursuit, harvest), axis=1)


def _return_conflicts(x, labels):
    F = sys.modules["cnet2_feat"]
    workers = {}
    bases = {}
    for slot in np.flatnonzero(x["tok_mask"][:F.OWN]):
        pos = int(x["tok_pos"][slot])
        if slot < 10:
            workers[int(slot)] = int(F.DEST[pos, int(labels[slot])])
        elif labels[slot] > 0:
            bases[int(slot)] = pos
    pairs = {(a,b) for a,da in workers.items() for b,db in workers.items() if a<b and da==db}
    return pairs | {(a,b) for a,da in workers.items() for b,db in bases.items() if da==db}


def _mixed_actions(x, aux):
    F = sys.modules["cnet2_feat"]
    enc = []
    for policy in _NETS:
        tokens, context = policy.encode(x, _HYBRID["style"])
        enc.append((policy,tokens,context,policy.lin(context,"h0")))
    baseline = _return_decode(x, aux, enc, False)
    if not np.any(_RETURN_THETA):
        return baseline
    candidate = _return_decode(x, aux, enc, True)
    protected = (np.arange(F.OWN)>=10) | (x["tok_kind"][:F.OWN]==1)
    if np.any(baseline[protected] != candidate[protected]):
        return baseline
    if np.any(baseline != candidate) and _return_conflicts(x,candidate) - _return_conflicts(x,baseline):
        return baseline
    return candidate
