#!/usr/bin/env python3
"""Part 2b (supplementary trip, strategy-group ruling 2026-11-15):
affine calibration of the m3s2r1 value leaf.

Same held-out set as Part 2 (ds filename order [32:64], 6,336 roots;
z + seat-oriented realized margin per root, extracted by Part 2's
fit_z.py pipeline and archived as heldout_z.npz, md5
17f26aa926f9a686e34ab6fba3bc331e). Joint OLS fit:

    margin = a * z + b

reported against Part 2's registered through-origin fit
M_eq = a0 * z (a0 = 3347.515730530027), which is recomputed from the
same arrays here as a pipeline self-check (must reproduce Part 2's
fit_results.json exactly). rootset32 is NOT touched by this fit.
"""
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SRC = os.path.join(DATA, "heldout_z.npz")
SRC_MD5 = "17f26aa926f9a686e34ab6fba3bc331e"
PART2_A = 3347.515730530027


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(resid, m):
    return {
        "resid_quantiles": {q: float(np.percentile(resid, q))
                            for q in (5, 25, 50, 75, 95)},
        "resid_mean": float(resid.mean()),
        "resid_std": float(resid.std()),
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "r2_centered": 1.0 - float((resid ** 2).sum())
        / float(((m - m.mean()) ** 2).sum()),
        "r2_uncentered": 1.0 - float((resid ** 2).sum())
        / float((m ** 2).sum()),
    }


def main():
    assert md5_file(SRC) == SRC_MD5, "heldout_z.npz md5 drift"
    d = np.load(SRC, allow_pickle=True)
    z = np.asarray(d["z"], dtype=np.float64)
    m = np.asarray(d["margin"], dtype=np.float64)
    n = len(z)
    assert n == 6336, n

    out = {"source_npz_md5": SRC_MD5, "n_roots": int(n)}

    # ---- self-check: through-origin fit must reproduce Part 2 exactly
    a0 = float((z * m).sum() / (z * z).sum())
    assert abs(a0 - PART2_A) < 1e-6, (a0, PART2_A)
    out["through_origin_recomputed"] = dict(
        {"a": a0}, **stats(m - a0 * z, m))
    out["self_check"] = "through-origin a reproduces Part 2 registered " \
                         "value 3347.515730530027 to <1e-6"

    # ---- affine OLS: margin = a*z + b
    X = np.stack([z, np.ones_like(z)], axis=1)
    coef, *_ = np.linalg.lstsq(X, m, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = m - (a * z + b)
    out["affine"] = dict({"a": a, "b": b}, **stats(resid, m))
    out["affine"]["pearson_z_margin"] = float(np.corrcoef(z, m)[0, 1])
    out["affine"]["a_deviation_vs_3000_pct"] = (a - 3000.0) / 3000.0 * 100.0
    out["affine"]["zero_crossing_z"] = -b / a
    out["z_stats"] = {"mean": float(z.mean()), "std": float(z.std()),
                      "min": float(z.min()), "max": float(z.max())}
    out["margin_stats"] = {"mean": float(m.mean()), "std": float(m.std()),
                            "min": float(m.min()), "max": float(m.max())}

    with open(os.path.join(DATA, "fit_affine_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
