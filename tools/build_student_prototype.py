"""build_student_prototype.py -- B-tier student weights with RANDOM init, in the
exact cnet2_np npz layout of the production teacher (cnet2_m3s2r1_heads_v2.npz).

Purpose: latency prototype only (stage-2 sequence per sign-off SS5: latency is
weight-value independent, so random weights prove section-4 reachability before
any training investment). NOT a trained student; must never be used as a teacher.

Layout mirrors cnet2_model.CNet2 state_dict key names/shapes for cfg B:
  C=48, NB=4, TA=2, D=96, L=2, NH=4, TNH=4, E=24, G=4
meta is a 0-d unicode JSON string, exactly like the teacher npz.
"""

import json

import numpy as np

CFG_B = dict(C=48, NB=4, D=96, NH=4, L=2, E=24, G=4, TA=2, TNH=4)
SEED = 20261117


def build(cfg=CFG_B, seed=SEED):
    rng = np.random.default_rng(seed)
    C, D, E = cfg["C"], cfg["D"], cfg["E"]
    NB, TA, L = cfg["NB"], cfg.get("TA", 0), cfg["L"]
    TNH, NH = cfg.get("TNH", 4), cfg["NH"]
    w = {}

    def lin(o, i, scale=None):
        s = scale if scale is not None else np.sqrt(2.0 / (i + o))
        return (rng.standard_normal((o, i)) * s).astype(np.float32)

    def conv(o, i, k=3):
        s = np.sqrt(2.0 / (i * k * k))
        return (rng.standard_normal((o, i, k, k)) * s).astype(np.float32)

    def zeros(n):
        return np.zeros(n, np.float32)

    def ones(n):
        return np.ones(n, np.float32)

    def small(shape, s=0.05):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    w["stem.weight"] = conv(C, 33); w["stem.bias"] = zeros(C)
    w["stem_n.weight"] = ones(C); w["stem_n.bias"] = zeros(C)
    for i in range(NB):
        p = f"blocks.{i}."
        w[p + "c1.weight"] = conv(C, C); w[p + "c1.bias"] = zeros(C)
        w[p + "n1.weight"] = ones(C); w[p + "n1.bias"] = zeros(C)
        w[p + "c2.weight"] = conv(C, C); w[p + "c2.bias"] = zeros(C)
        w[p + "n2.weight"] = ones(C); w[p + "n2.bias"] = zeros(C)
    for j in range(TA):
        p = f"tattn.{j}."
        w[p + "n1.weight"] = ones(C); w[p + "n1.bias"] = zeros(C)
        w[p + "qkv.weight"] = lin(3 * C, C); w[p + "qkv.bias"] = zeros(3 * C)
        w[p + "out.weight"] = lin(C, C); w[p + "out.bias"] = zeros(C)
        w[p + "rel.weight"] = small((225, TNH))
        w[p + "n2.weight"] = ones(C); w[p + "n2.bias"] = zeros(C)
        w[p + "f1.weight"] = lin(2 * C, C); w[p + "f1.bias"] = zeros(2 * C)
        w[p + "f2.weight"] = lin(C, 2 * C); w[p + "f2.bias"] = zeros(C)
    w["aux.weight"] = conv(1, C, k=1); w["aux.bias"] = zeros(1)
    w["tok1.weight"] = lin(D, 89); w["tok1.bias"] = zeros(D)
    w["tok2.weight"] = lin(D, D); w["tok2.bias"] = zeros(D)
    w["fuse.weight"] = lin(D, C + D); w["fuse.bias"] = zeros(D)
    for l in range(L):
        p = f"attn.{l}."
        w[p + "qkv.weight"] = lin(3 * D, D); w[p + "qkv.bias"] = zeros(3 * D)
        w[p + "out.weight"] = lin(D, D); w[p + "out.bias"] = zeros(D)
        w[p + "rel.weight"] = small((225, NH))
        w[p + "n1.weight"] = ones(D); w[p + "n1.bias"] = zeros(D)
        w[p + "f1.weight"] = lin(2 * D, D); w[p + "f1.bias"] = zeros(2 * D)
        w[p + "f2.weight"] = lin(D, 2 * D); w[p + "f2.bias"] = zeros(D)
        w[p + "n2.weight"] = ones(D); w[p + "n2.bias"] = zeros(D)
    w["sc1.weight"] = lin(64, 20); w["sc1.bias"] = zeros(64)
    w["sc2.weight"] = lin(64, 64); w["sc2.bias"] = zeros(64)
    w["teach.weight"] = small((8, 16), 0.1)
    w["ctx.weight"] = lin(D, C + D + 64 + 16); w["ctx.bias"] = zeros(D)
    w["value.weight"] = lin(1, D); w["value.bias"] = zeros(1)
    w["utility.weight"] = lin(1, D); w["utility.bias"] = zeros(1)
    w["wdl.weight"] = lin(3, D); w["wdl.bias"] = zeros(3)
    w["margin.weight"] = lin(1, D); w["margin.bias"] = zeros(1)
    w["h0.weight"] = lin(D, D); w["h0.bias"] = zeros(D)
    w["din.weight"] = lin(D, 2 * D + 9 + 5 * 15); w["din.bias"] = zeros(D)
    w["aemb.weight"] = small((9, D), 0.1)
    w["gru.weight_ih"] = lin(3 * D, D); w["gru.weight_hh"] = lin(3 * D, D)
    w["gru.bias_ih"] = zeros(3 * D); w["gru.bias_hh"] = zeros(3 * D)
    w["whead.weight"] = lin(5, D); w["whead.bias"] = zeros(5)
    w["bhead.weight"] = lin(3, D); w["bhead.bias"] = zeros(3)
    w["U.weight"] = lin(E, D); w["U.bias"] = zeros(E)
    w["V.weight"] = lin(E, 15); w["V.bias"] = zeros(E)
    n = int(sum(int(v.size) for v in w.values()))
    meta = {"cfg": cfg, "epoch": -1, "params": n,
            "identity": {"kind": "small_teacher_prototype_random",
                         "purpose": "latency prototype only -- random weights, NOT trained, never a teacher",
                         "distilled_from": {"teacher_code_md5": "c6ade5ced42e2c54a77c1499163aa151",
                                            "teacher_weights_md5": "06a03d9cfeec239af3065a49fefec382"},
                         "seed": seed}}
    w["meta"] = np.str_(json.dumps(meta))
    return w, n


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "student_B_prototype.npz"
    w, n = build()
    np.savez(out, **w)
    print(f"wrote {out}: {len(w) - 1} arrays + meta, params={n}")
