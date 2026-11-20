def _identity_uniforms(seed, root_roster):
    """A common random draw per entity, shared by all simulated alternatives."""
    import hashlib
    root_ids = list(root_roster[1]) + list(root_roster[2])
    draws = dict(zip(root_ids, np.random.RandomState(seed).random_sample(len(root_ids))))

    def choose(uid, probabilities):
        if uid not in draws:
            # Newborns can exist in just one branch. They must not consume a survivor's draw.
            key = str(seed).encode("ascii") + b"/" + uid.encode("utf-8")
            bits = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
            draws[uid] = (bits >> 11) * (2.0 ** -53)
        # RandomState.choice(p=...) accumulates normalized probabilities in double precision.
        cdf = np.cumsum(probabilities, dtype=np.float64)
        cdf /= cdf[-1]
        return int(cdf.searchsorted(draws[uid], side="right"))

    return choose
