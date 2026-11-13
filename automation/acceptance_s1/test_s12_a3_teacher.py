"""S1-2 A3 teacher: guard-on/off comparison, CachePolicy parity, guard-on smoke.

50 root positions from m3s2 replays (steps 40-240). Small teacher params
(K=4, M=2, H=2) so the comparison finishes on CPU; the guard wiring under
test is identical at full K/M. Differences are REPORTED, not judged.
Run: <venv python> test_s12_a3_teacher.py  (CPU only)
"""
import gzip
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cnet2_feat as F  # noqa: E402
import cnet2_np as NP  # noqa: E402
from cnet2_search import CachePolicy  # noqa: E402
from eval_v53 import CFG  # noqa: E402
from fastsim import FastCarbon  # noqa: E402
import stage0_teacher as T  # noqa: E402

REPLAYS = sorted(os.path.join(HERE, "replays", f) for f in os.listdir(os.path.join(HERE, "replays"))
                 if f.endswith(".json.gz"))


def roots(n_target=50):
    """(root_full, me, x_me, aux_me, x_opp, aux_opp, feat_me, feat_opp) per root.
    root_full is snapshotted at capture time (the live sim keeps stepping)."""
    import types
    out = []
    for rp in REPLAYS:
        d = json.load(gzip.open(rp))
        st = d["steps"]
        sim = FastCarbon(CFG).reset(seed=1, full_carbon=st[0][0]["observation"]["full_carbon"])
        for t in range(299):
            if 40 <= t <= 240 and t % 17 == 0:
                me = 0
                fm = F.Feat(me)
                fo = F.Feat(1 - me)
                x_me = fm.observe(sim.obs(me))
                x_opp = fo.observe(sim.obs(1 - me))
                out.append((sim.full_observation(0), me, x_me, fm.aux, x_opp, fo.aux,
                            types.SimpleNamespace(seat=me, own_uid=fm.own_uid),
                            types.SimpleNamespace(seat=1 - me, own_uid=fo.own_uid)))
                if len(out) >= n_target:
                    return out
            sim.step(st[t + 1][0].get("action") or {}, st[t + 1][1].get("action") or {})
    return out


def main():
    w = dict(np.load(os.path.join(HERE, "heads_v2.npz")))
    pol_np = NP.Policy(w)
    pol_cache = CachePolicy(w)
    R = roots(50)
    print(f"roots: {len(R)}")
    # Fixture identity lock (Codex R2 P3): the canonical fixture yields 50
    # roots; a shrunken or duplicated fixture must not silently pass.
    assert len(R) >= 40, f"fixture shrank: only {len(R)} roots"
    K, M, H = 4, 2, 2

    # --- CachePolicy parity (guard off): NP.Policy vs CachePolicy must agree exactly.
    max_dq = 0.0
    t0 = time.time()
    for (root_full, me, xm, am, xo, ao, fm, fo) in R:
        sim = FastCarbon(CFG)
        sim.load_observation(root_full)
        r1 = T.evaluate_root(sim, me, pol_np, xm, am, xo, ao, fm, fo,
                             teacher=1, K=K, M=M, H=H, guard=None)
        r2 = T.evaluate_root(sim, me, pol_cache, xm, am, xo, ao, fm, fo,
                             teacher=1, K=K, M=M, H=H, guard=None)
        max_dq = max(max_dq, float(np.abs(r1["q"] - r2["q"]).max()),
                     float(np.abs(r1["Q_mean"] - r2["Q_mean"]).max()))
        assert all(np.array_equal(a, b) for a, b in zip(r1["arms"], r2["arms"])), "arms differ"
        # Hardened (Codex review P3): NaN must not hide inside max()
        # reductions, and q must be a normalised distribution.
        for r in (r1, r2):
            assert np.isfinite(r["q"]).all() and np.isfinite(r["Q_mean"]).all(), \
                "non-finite teacher output"
            assert abs(float(r["q"].sum()) - 1.0) < 1e-6, "q not normalised"
    print(f"CachePolicy parity (guard off): max |dq|={max_dq:.3e} over {len(R)} roots "
          f"({time.time()-t0:.0f}s)")
    assert max_dq == 0.0, "CachePolicy parity FAILED (must be exact)"

    # --- guard on/off comparison (report only) + guard-on smoke.
    q_on, q_off, qm_on, qm_off = [], [], [], []
    gm_on = gr_on = 0
    t0 = time.time()
    for (root_full, me, xm, am, xo, ao, fm, fo) in R:
        sim = FastCarbon(CFG)
        sim.load_observation(root_full)
        r_on = T.evaluate_root(sim, me, pol_np, xm, am, xo, ao, fm, fo,
                               teacher=1, K=K, M=M, H=H, guard="on")
        r_off = T.evaluate_root(sim, me, pol_np, xm, am, xo, ao, fm, fo,
                                teacher=1, K=K, M=M, H=H, guard=None)
        # Hardened (Codex R2 P3): the finite/shape/normalisation checks
        # apply to the guard-ON path too, not only to the parity loop.
        for r in (r_on, r_off):
            assert np.isfinite(r["q"]).all() and np.isfinite(r["Q_mean"]).all(), \
                "non-finite teacher output (guard-on/off smoke)"
            assert (r["q"] >= 0).all() and abs(float(r["q"].sum()) - 1.0) < 1e-6, \
                "q not a normalised distribution (guard-on/off smoke)"
            assert len(r["arms"]) == len(r["q"]) == len(r["Q_mean"])
        q_on.append(r_on["q"])
        q_off.append(r_off["q"])
        qm_on.append(r_on["Q_mean"])
        qm_off.append(r_off["Q_mean"])
        gm_on += r_on["guard"]["guard_moves"]
        gr_on += r_on["guard"]["guard_recruit"]
    q_on = np.array(q_on)
    q_off = np.array(q_off)
    qm_on = np.array(qm_on)
    qm_off = np.array(qm_off)
    n_diff = int((np.abs(q_on - q_off).max(axis=1) > 1e-12).sum())
    print(f"guard-on smoke: {len(R)} roots OK, guard_moves={gm_on}, guard_recruit={gr_on} "
          f"({time.time()-t0:.0f}s, K={K} M={M} H={H})")
    print("guard on/off (report only, same 50 roots):")
    print(f"  roots with any q change: {n_diff}/{len(R)}")
    print(f"  mean |q_on - q_off| = {np.abs(q_on - q_off).mean():.4f}, "
          f"max = {np.abs(q_on - q_off).max():.4f}")
    print(f"  Q_mean on  = {qm_on.mean():.4f} +- {qm_on.std():.4f}")
    print(f"  Q_mean off = {qm_off.mean():.4f} +- {qm_off.std():.4f}")
    print(f"  mean(Q_on - Q_off) per root: mean={np.mean(qm_on.mean(1) - qm_off.mean(1)):+.4f}, "
          f"min={np.min(qm_on.mean(1) - qm_off.mean(1)):+.4f}, "
          f"max={np.max(qm_on.mean(1) - qm_off.mean(1)):+.4f}")
    print("A3 teacher PASS (smoke ran; parity exact; on/off differences reported)")


if __name__ == "__main__":
    main()
