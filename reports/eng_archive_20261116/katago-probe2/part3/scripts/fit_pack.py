#!/usr/bin/env python3
"""Part 3 refit steps 2-3 (strategy-group ruling 2026-11-15 13:34,
sec.3 disposition (iii)): dz distribution + pack-space affine refit.

Inputs: part3/data/heldout_pack_z.npz (extract_pack_z.py): per held-out
root (same 6,336 as Part 2/2B), z_fp (Part 2 fit_z.py pipeline value,
from heldout_z.npz 17f26aa9), z_pack (deployed pack path), margin
(ds meta scores, seat-oriented).

Step 2: dz = z_pack - z_fp over all 6,336 roots -- mean / SD /
quantiles (5/25/50/75/95) / Pearson(dz, z_fp) (+ dz~z_fp OLS slope and
intercept, to settle whether the shift is near-constant).

Step 3: OLS margin = a' * z_pack + b' (float64 lstsq, same conventions
as part2b fit_affine.py), reported side by side with the fp-space fit.
MARGIN' = 1.0 * pack-space residual SD, rounded to 0.1 (the accepted
rule: 1.0x sigma; the tail belongs to the quality gate's negative-
fraction line). Self-checks: the fp arrays must reproduce the Part 2
through-origin a (3347.515730530027, <1e-6) and the Part 2B affine pair
(3687.9177926610573, -638.5907028300794, <1e-9) before any pack number
is reported.
"""
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SRC = os.path.join(DATA, "heldout_pack_z.npz")
FP_A0 = 3347.515730530027
FP_A = 3687.9177926610573
FP_B = -638.5907028300794
FP_SD = 625.5417075364155


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


def affine(z, m):
    X = np.stack([z, np.ones_like(z)], axis=1)
    coef, *_ = np.linalg.lstsq(X, m, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    out = dict({"a": a, "b": b}, **stats(m - (a * z + b), m))
    out["pearson_z_margin"] = float(np.corrcoef(z, m)[0, 1])
    out["a_deviation_vs_3000_pct"] = (a - 3000.0) / 3000.0 * 100.0
    out["zero_crossing_z"] = -b / a
    return out


def main():
    d = np.load(SRC, allow_pickle=True)
    z_fp = np.asarray(d["z_fp"], dtype=np.float64)
    z_pk = np.asarray(d["z_pack"], dtype=np.float64)
    m = np.asarray(d["margin"], dtype=np.float64)
    n = len(z_fp)
    assert n == 6336 and len(z_pk) == n and len(m) == n
    assert np.isfinite(z_pk).all()

    out = {"source_npz": os.path.basename(SRC),
           "source_npz_md5": md5_file(SRC), "n_roots": int(n)}

    # ---- self-checks on the fp side (must reproduce registered values)
    a0 = float((z_fp * m).sum() / (z_fp * z_fp).sum())
    assert abs(a0 - FP_A0) < 1e-6, (a0, FP_A0)
    fp = affine(z_fp, m)
    assert abs(fp["a"] - FP_A) < 1e-9 and abs(fp["b"] - FP_B) < 1e-9, fp
    assert abs(fp["resid_std"] - FP_SD) < 1e-9, fp["resid_std"]
    out["self_check"] = ("fp side reproduces Part 2 through-origin a "
                         "(<1e-6) and Part 2B affine pair + resid SD "
                         "(<1e-9) from the paired arrays")
    out["fp_affine"] = fp

    # ---- step 2: dz distribution over all 6,336 roots
    dz = z_pk - z_fp
    X = np.stack([z_fp, np.ones_like(z_fp)], axis=1)
    coef, *_ = np.linalg.lstsq(X, dz, rcond=None)
    slope, icept = float(coef[0]), float(coef[1])
    dz_fit_resid = dz - (slope * z_fp + icept)
    out["dz"] = {
        "mean": float(dz.mean()),
        "std": float(dz.std()),
        "min": float(dz.min()),
        "max": float(dz.max()),
        "quantiles": {q: float(np.percentile(dz, q))
                      for q in (5, 25, 50, 75, 95)},
        "pearson_dz_zfp": float(np.corrcoef(dz, z_fp)[0, 1]),
        "pearson_dz_zpack": float(np.corrcoef(dz, z_pk)[0, 1]),
        "pearson_zpack_zfp": float(np.corrcoef(z_pk, z_fp)[0, 1]),
        "ols_dz_on_zfp": {
            "slope": slope, "intercept": icept,
            "resid_std": float(dz_fit_resid.std()),
            "r2": 1.0 - float((dz_fit_resid ** 2).sum())
            / float(((dz - dz.mean()) ** 2).sum()),
        },
        "frac_within_0.05_of_mean": float(
            (np.abs(dz - dz.mean()) <= 0.05).mean()),
        "frac_within_0.10_of_mean": float(
            (np.abs(dz - dz.mean()) <= 0.10).mean()),
    }
    out["zpack_stats"] = {"mean": float(z_pk.mean()),
                          "std": float(z_pk.std()),
                          "min": float(z_pk.min()),
                          "max": float(z_pk.max())}
    out["zfp_stats"] = {"mean": float(z_fp.mean()), "std": float(z_fp.std()),
                        "min": float(z_fp.min()), "max": float(z_fp.max())}

    # ---- step 3: pack-space affine refit + MARGIN'
    pk = affine(z_pk, m)
    out["pack_affine"] = pk
    sigma = pk["resid_std"]
    margin_new = round(1.0 * sigma, 1)
    out["margin_prime"] = {
        "sigma_pack_unrounded": sigma,
        "rule": "1.0 * pack-space residual SD, rounded to 0.1 "
                "(accepted rule; tail governed by the quality gate)",
        "value": margin_new,
        "fp_predecessor": 625.5,
        "differs_from_625_5": bool(margin_new != 625.5),
    }

    with open(os.path.join(DATA, "fit_pack_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
