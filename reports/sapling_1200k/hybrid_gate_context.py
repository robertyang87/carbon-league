"""Indices for a role/phase/threat/cargo mixing table, from visible state only."""
import numpy as np


def hybrid_categories(kinds, positions, mask, cargo, steps, distances):
    """Rows are observer-relative; enemy workers are tokens 12..21."""
    own = positions[:, :12].astype(np.int64)
    enemy = positions[:, 12:22].astype(np.int64)
    nearby = ((distances[own[:, :, None], enemy[:, None, :]] <= 2)
              & mask[:, None, 12:22]).any(-1)
    role = kinds[:, :12].astype(np.int64)
    loaded = (role == 0) & (cargo[:, :12] >= 30)
    phase = (steps >= 100).astype(np.int64) + (steps >= 200)
    return ((role * 3 + phase[:, None]) * 2 + nearby) * 2 + loaded
