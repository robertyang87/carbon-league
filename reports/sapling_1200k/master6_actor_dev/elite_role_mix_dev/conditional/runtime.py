def _role_residual_features(role, step, unit, move, legal, first, second):
    # All five rows refer to the same acting slot and live joint prefix.
    first = np.where(legal, first, -1e9)
    second = np.where(legal, second, -1e9)
    a = np.clip((first - np.max(first)) / 3., -3., 0.)
    b = np.clip((second - np.max(second)) / 3., -3., 0.)
    m = np.clip(move.astype(np.float32), -2., 2.)
    phase = np.float32(step / 299.)
    cargo = np.float32(unit[3])
    # Base columns identify NONE / recruit collector / recruit planter.
    base_choice = np.eye(5, dtype=np.float32)[:,:3] * (role == 2)
    stay = np.array([1., 0., 0., 0., 0.], dtype=np.float32)[:, None]
    actor = np.stack([a, b, a*b, a*cargo, b*cargo, a*phase, b*phase], axis=1)
    return np.concatenate([actor, m, m*cargo, m*phase, base_choice, base_choice*phase, stay], axis=1)


def _role_residual(x, slot, unit, move, legal, first, second):
    step = 299 - x["left"]
    role = int(x["tok_kind"][slot])
    if step == 0 or np.argmax(np.where(legal, first, -np.inf)) == np.argmax(np.where(legal, second, -np.inf)):
        return 0.
    features = _role_residual_features(role, step, unit, move, legal, first, second)
    return 2.*np.tanh(features @ _ROLE_THETA[role] / 2.)
