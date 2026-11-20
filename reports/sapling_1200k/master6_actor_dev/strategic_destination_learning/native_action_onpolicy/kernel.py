"""Probability and log-probability derivative of the issued physical team order."""
import numpy as np


def physical(menu, count, weight):
    if menu is None:
        if count != 1:
            raise ValueError("Deterministic step has multiple orders")
        return np.ones(1), np.zeros((1, len(weight)))
    hidden = np.asarray(menu["hidden"], dtype=np.float32)
    logits = np.asarray(menu["logits"], dtype=np.float32)
    routes = np.asarray(menu["route_to_order"], dtype=np.int64)
    base = menu["base_index"]
    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    if (hidden.shape != (len(routes), len(weight)) or logits.shape != (len(routes),) or not 0 <= base < count or
        np.any((routes < 0) | (routes >= count)) or not np.all(np.isfinite(hidden)) or not np.all(np.isfinite(logits))):
        raise ValueError("Malformed physical policy features")
    values = (logits + hidden @ weight).astype(np.float64)
    hazards = np.exp(values - values.max())
    probabilities = hazards / hazards.sum()
    masses = np.bincount(routes, weights=0.5 * probabilities, minlength=count)
    masses[base] += 0.5
    if np.any(masses <= 0) or abs(masses.sum() - 1) > 1e-12:
        raise ValueError("Physical distribution lost support")
    centers = hidden - probabilities @ hidden
    derivative = np.zeros((count,len(weight)), dtype=np.float64)
    np.add.at(derivative, routes, 0.5 * probabilities[:,None] * centers)
    return masses, derivative / masses[:,None]
