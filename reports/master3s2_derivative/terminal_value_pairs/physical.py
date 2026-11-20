"""Public, seat-relative terminal cargo exposures on the 15×15 torus, in units of 3,000."""
import numpy as np


def distance(cell, base):
    qr, qc = divmod(cell, 15)
    br, bc = divmod(base, 15)
    dr, dc = abs(qr - br), abs(qc - bc)
    return min(dr, 15 - dr) + min(dc, 15 - dc)


def public_features(players, step, me):
    if me not in (0, 1) or len(players) != 2 or not 0 <= step <= 299:
        raise ValueError("Expected both players, a valid seat and native board step 0..299")
    remaining = 299 - step
    unreachable = transit = 0.0
    for seat, (_, bases, workers, _) in enumerate(players):
        sign = -1.0 if seat == me else 1.0
        destinations = tuple(bases.values())
        for cell, carried, role in workers.values():
            if role != "COLLECTOR" or carried == 0:
                continue
            d = min((distance(cell, base) for base in destinations), default=remaining + 1)
            if remaining == 0 or d > remaining:
                unreachable += sign * carried
            else:
                transit += sign * carried * d / remaining
    return np.array([unreachable / 3000., transit / 3000.], dtype=np.float64)
