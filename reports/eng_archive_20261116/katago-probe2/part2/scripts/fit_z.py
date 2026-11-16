#!/usr/bin/env python3
"""Part 2 (PREREG_SEARCH §2): z extraction + through-origin fit.

z = value-head pre-activation of the m3s2r1 actor itself
(cnet2_m3s2r1L.npz, tanh head): forward via the repo adapter path
(rl2_ppo_cnet2.Cnet2Adapter.make_model), v = tanh(z) is the model's
value output, z = atanh(v) in float64. Batch assembly mirrors the
value-diag common.batch rules verbatim.

Validation first: the same extraction on the registered rootset32
roots.npz rows of the first 4 games (792 roots) must reproduce the
construction receipt's identity measurement Pearson(u_hat, U)=0.849
(u_hat=(tanh(z)+1)/2, U from rootset32 scores by the platform
formula). Then the fit on the held-out set: margin_seat (ds meta
scores, seat-oriented) regressed on z through the origin,
M_eq = a * z; registered value = fitted a.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

CODE = "/home/arm/junyang/carbon/code"
sys.path.insert(0, CODE)

WORK = "/home/arm/junyang/katago_probe_scratch/part2"
ARCH = "/home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib"
WEIGHTS = "/home/arm/junyang/s26stage/data/weights/cnet2_m3s2r1L.npz"
WEIGHTS_MD5 = "b6db08c9b076257c54817377bb93130d"


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def batch(z, idx):
    """value-diag common.batch, verbatim rules."""
    import torch
    import cnet2_feat as F
    d = {}
    for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_mask",
              "tok_cargo", "hist_cell", "hist_cargo", "hist_move",
              "uf", "mf", "legal", "y", "teacher"):
        a = z[k][idx]
        if a.dtype == np.float16:
            a = a.astype(np.float32)
        d[k] = torch.from_numpy(np.ascontiguousarray(a))
    B = len(idx)
    d["maps"] = (d["maps"].float() / 255.0).reshape(B, F.P, F.S, F.S)
    for k in ("tok_pos", "tok_kind", "hist_cell", "hist_move", "y",
              "teacher"):
        d[k] = d[k].long()
    d["tok_mask"] = d["tok_mask"].bool()
    d["legal"] = d["legal"].bool()
    return d


def u_of_margin(m):
    w = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-float(m) / 1000.0))


def forward_z(model, z, idx_all, device, mb=512):
    import torch
    vs = []
    with torch.no_grad():
        for i in range(0, len(idx_all), mb):
            idx = idx_all[i:i + mb]
            b = {kk: vv.to(device) for kk, vv in batch(z, idx).items()}
            _lg, v, _aux = model(b)
            vs.append(v.double().cpu().numpy())
    v = np.concatenate(vs).ravel()
    v = np.clip(v, -(1 - 1e-12), 1 - 1e-12)
    return np.arctanh(v)


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def main():
    import torch
    from rl2_ppo_cnet2 import Cnet2Adapter
    assert md5_file(WEIGHTS) == WEIGHTS_MD5, "weights md5 drift"
    d = np.load(WEIGHTS, allow_pickle=True)
    weights = {k: d[k] for k in d.files}
    model = Cnet2Adapter().make_model(weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    out = {"weights_md5": WEIGHTS_MD5, "device": device}

    # ---- validation: first 4 games of registered roots.npz (792 roots)
    rz = np.load(os.path.join(ARCH, "roots.npz"), allow_pickle=True)
    rs = json.load(open(os.path.join(ARCH, "rootset32.json")))
    gk = np.asarray([str(x) for x in rz["game_key"]])
    seat = np.asarray(rz["seat"]).astype(int)
    first4 = [g["key"] for g in rs[:4]]
    sel = np.array([i for i in range(len(gk)) if gk[i] in first4])
    assert len(sel) == 792, len(sel)
    zv = forward_z(model, rz, sel, device)
    sc = {g["key"]: g["scores"] for g in rs}
    U = np.array([u_of_margin(sc[gk[i]][seat[i]] - sc[gk[i]][1 - seat[i]])
                  for i in sel])
    uhat = (np.tanh(zv) + 1.0) / 2.0
    out["validation_792"] = {
        "n": int(len(sel)),
        "pearson_uhat_U": pearson(uhat, U),
        "registered_reference": 0.849,
        "uhat_mean": float(uhat.mean()), "uhat_std": float(uhat.std()),
        "uhat_min": float(uhat.min()), "uhat_max": float(uhat.max()),
    }
    print("validation:", out["validation_792"], flush=True)

    # ---- held-out fit
    hz = np.load(os.path.join(WORK, "heldout.npz"), allow_pickle=True)
    man = json.load(open(os.path.join(WORK, "heldout_manifest.json")))
    hsc = {g["key"]: g["scores"] for g in man}
    hgk = np.asarray([str(x) for x in hz["game_key"]])
    hseat = np.asarray(hz["seat"]).astype(int)
    hstep = np.asarray(hz["step_arr"]).astype(int)
    idx = np.arange(len(hgk))
    zh = forward_z(model, hz, idx, device)
    m = np.array([hsc[hgk[i]][hseat[i]] - hsc[hgk[i]][1 - hseat[i]]
                  for i in idx], dtype=np.float64)
    a = float((zh * m).sum() / (zh * zh).sum())
    resid = m - a * zh
    resid3000 = m - 3000.0 * zh
    ss_res = float((resid ** 2).sum())
    out["fit"] = {
        "n_roots": int(len(idx)),
        "a": a,
        "deviation_vs_3000_pct": (a - 3000.0) / 3000.0 * 100.0,
        "pearson_z_margin": pearson(zh, m),
        "r2_uncentered": 1.0 - ss_res / float((m ** 2).sum()),
        "r2_centered": 1.0 - ss_res / float(((m - m.mean()) ** 2).sum()),
        "resid_quantiles": {q: float(np.percentile(resid, q))
                            for q in (5, 25, 50, 75, 95)},
        "resid_mean": float(resid.mean()), "resid_std": float(resid.std()),
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "at_3000": {
            "resid_quantiles": {q: float(np.percentile(resid3000, q))
                                for q in (5, 25, 50, 75, 95)},
            "mae": float(np.abs(resid3000).mean()),
            "rmse": float(np.sqrt((resid3000 ** 2).mean())),
        },
        "z_stats": {"mean": float(zh.mean()), "std": float(zh.std()),
                    "min": float(zh.min()), "max": float(zh.max())},
        "margin_stats": {"mean": float(m.mean()), "std": float(m.std()),
                         "min": float(m.min()), "max": float(m.max())},
    }
    # per-game a (stability side-read)
    per_game = {}
    for g in man:
        s = hgk == g["key"]
        if (zh[s] ** 2).sum() > 0:
            per_game[g["key"]] = float((zh[s] * m[s]).sum()
                                       / (zh[s] ** 2).sum())
    av = np.array(list(per_game.values()))
    out["fit"]["per_game_a"] = {"min": float(av.min()),
                                "median": float(np.median(av)),
                                "max": float(av.max())}
    np.savez(os.path.join(WORK, "heldout_z.npz"), z=zh, margin=m,
             game_key=hgk, seat=hseat, step=hstep)
    with open(os.path.join(WORK, "fit_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["fit"], indent=1), flush=True)


if __name__ == "__main__":
    main()
