"""Spatial carbon residual from the active seat's original public Sapling tracker."""
import numpy as np


def _carbon_inputs(x, feat, F):
    planes = x["maps"].reshape(-1, 225).astype(np.float32) / np.float32(255)
    step = 299 - x["left"]
    order = F.CANON if feat.seat else F.GRID
    neighbors = F.DEST[:, 1:]
    mirrors = F.MIRROR
    own_tree = planes[16]
    enemy_tree = planes[17]
    eligible = (planes[2] == 0) & (planes[14:18].sum(axis=0) == 0)
    # Reflect the *measurements* and their ages. The learned residual decides
    # which traces remain informative after asymmetric collection or tree growth.
    last_value = (feat.last[order] / 100).astype(np.float32)
    last_time = feat.last_step[order]
    mirror_last = last_time[mirrors]
    latest = mirror_last.argmax(axis=1)
    source = mirrors[F.GRID, latest]
    has_reading = last_time[source] >= 0
    last_age = np.where(has_reading, step - last_time[source], 300)
    carbon = np.zeros(225, np.float32)
    first = np.full(225, 300, np.int32)
    for physical, (value, when) in feat.inf.items():
        canonical = order[physical]
        carbon[canonical] = value / 100
        first[canonical] = when
    first_id = mirrors[F.GRID, first[mirrors].argmin(axis=1)]
    first_time = first[first_id]
    has_gain = first_time < 300
    elapsed = np.where(has_gain, step - first_time, 300)
    rich = [
        planes[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 20, 21, 24, 28, 29, 30)
    ]
    for field in (planes[0], last_value, planes[2], own_tree + enemy_tree):
        rich.append(field[neighbors].mean(axis=1))
    rich.extend([
        last_value[source], has_reading, last_age / 300,
        last_value[mirrors].max(axis=1), planes[2][mirrors].mean(axis=1),
        planes[0][mirrors].mean(axis=1), (own_tree + enemy_tree)[mirrors].mean(axis=1),
        (planes[29] + planes[30])[mirrors].mean(axis=1),
        carbon, first < 300, first / 300, carbon[first_id], has_gain, first_time / 300,
        elapsed / 300, np.minimum(elapsed, 50) / 50, (first[mirrors] < 300).mean(axis=1),
        np.full(225, step / 300, np.float32), np.full(225, min(step, 100) / 100, np.float32),
    ])
    return np.stack(rich, axis=1).astype(np.float32), eligible


def _carbon_view(x, feat, F, weights):
    features, eligible = _carbon_inputs(x, feat, F)
    if not eligible.any():
        return x
    base = features[:, 0]
    if len(weights) == 1:
        change = weights[0]
    else:
        w, bias, readout, intercept = weights
        change = np.maximum(features @ w + bias, 0) @ readout + intercept
    maps = x["maps"].copy()
    corrected = F.q8(base + change)
    maps[0].reshape(-1)[eligible] = corrected[eligible]
    return dict(x, maps=maps)
