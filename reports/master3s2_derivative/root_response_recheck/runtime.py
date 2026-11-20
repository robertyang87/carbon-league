class _AntitheticResponse:
    """Complement the native categorical uniforms, keeping draw order and marginals."""

    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def choice(self, count, p):
        assert len(p) == count
        cumulative = np.cumsum(p, dtype=np.float64)
        u = 1.0 - self.rng.random_sample()
        return min(count-1, int(np.searchsorted(cumulative, u * cumulative[-1], side='right')))
