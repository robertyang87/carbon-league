"""Teacher v2 acceptance (spec SS15 item 3): the two-world experiment.

Standard: when two worlds differ ONLY in carbon cells the acting seat has
no view of, the teacher's Q must not change (beyond sampling noise -- v2
never reads the true carbon map, so the expected difference is exactly 0).

Runs from a repo checkout, no machine-side fixtures:
  python3 automation/acceptance_s1/test_teacher_v2_twoworld.py [WEIGHTS.npz] [--canon]
WEIGHTS defaults to data/weights/cnet2_b128.npz. Weights without the Stage 0
utility head get a seeded random head in memory (the test is about
information flow, not value quality; magnitudes under such a head are not
calibrated). --canon runs K=16, M=8, H=2 (about 20 s per evaluation on one
core); the default K=8, M=4, H=2 exercises the same code paths faster.

Checks per root position (three positions, steps 60 / 120 / 200):
  A. v2 (belief="sample"), worlds differing in cells invisible to BOTH seats
     (the review's construction): max|dQ| == 0, same arms, same q.
  B. v2, worlds differing in every cell not visible to the acting seat
     (same root inputs): max|dQ| == 0.
  C. control, belief="oracle" on construction A: max|dQ| > 0 -- proves the
     experiment has the power to detect a dependence.
  D. determinism: same world twice -> identical Q.
  E. root Feats are not mutated by the teacher (branches use copies).
Plus, on an early position (step 8, most cells still unknown): unit checks
of belief_carbon (known cells = belief, orbit-consistent samples, scenario
streams differ) and the two-world check with every unknown cell changed;
and the argument guards.
Exit 0 iff everything passes.
"""
import copy
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import cnet2_feat as F  # noqa: E402
from cnet2_search import CachePolicy, feat_copy  # noqa: E402
from eval_v53 import CFG  # noqa: E402
from fastsim import FastCarbon  # noqa: E402
import stage0_teacher as T  # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)


def load_policy(path):
    w = dict(np.load(path))
    if "utility.weight" not in w:
        rng = np.random.default_rng(20261113)
        D = w["value.weight"].shape[1]
        w["utility.weight"] = rng.normal(0, 0.3, (1, D)).astype(np.float32)
        w["utility.bias"] = rng.normal(0, 0.3, (1,)).astype(np.float32)
        print(f"NOTE weights {os.path.basename(path)} have no utility head: seeded random head added in memory", flush=True)
    return CachePolicy(w)


def root_position(pol, seed, stop):
    """Sampled self-play (temp 1.0, breaks the mirror symmetry) up to `stop`;
    both Feats have observed the root step."""
    rng = np.random.default_rng(seed)
    sim = FastCarbon(CFG).reset(seed=seed)
    fs = [F.Feat(0), F.Feat(1)]
    for _ in range(stop):
        acts = []
        for p in (0, 1):
            x = fs[p].observe(sim.obs(p))
            y, _, _ = pol.act(x, fs[p].aux, teacher=1, sample_rng=rng, temp=1.0)
            acts.append(fs[p].to_actions(y))
        sim.step(acts[0], acts[1])
    xs = [fs[0].observe(sim.obs(0)), fs[1].observe(sim.obs(1))]
    return sim, fs, xs


def visible(sim, p):
    v = np.zeros(F.N, bool)
    for w in sim.workers.values():
        if w.player != p:
            continue
        if w.is_collector:
            v[w.pos] = True
        else:
            for nb in sim._nbr4[w.pos]:
                v[nb] = True
    return v


def other_world(sim, cells):
    """Same public state, different carbon on `cells`."""
    b = FastCarbon(CFG)
    st = sim.get_state()
    full = sim.full_observation(0)
    full["tid_counter"] = st["tid_counter"]; full["wid_counter"] = st["wid_counter"]
    b.load_observation(full)
    for c in cells:
        b.carbon[c] = 0.0 if sim.carbon[c] > 20 else 80.0
    return b


def feat_sig(f):
    return (f.est.tobytes(), f.last.tobytes(), f.last_step.tobytes(), int(f.step),
            repr(sorted((k, tuple(v)) for k, v in f.hist.items())))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    canon = "--canon" in sys.argv
    wpath = args[0] if args else os.path.join(REPO, "data", "weights", "cnet2_b128.npz")
    K, M, H = (16, 8, 2) if canon else (8, 4, 2)
    pol = load_policy(wpath)
    print(f"CONFIG weights={wpath} K={K} M={M} H={H} teacher_version={T.TEACHER_VERSION}", flush=True)
    ns = lambda f: types.SimpleNamespace(seat=f.seat, own_uid=list(f.own_uid))

    for seed, stop in ((21, 60), (22, 120), (23, 200)):
        sim, fs, xs = root_position(pol, seed, stop)
        me = 0
        tag = f"seed{seed}/step{stop}"

        def run(world, belief, feats=True):
            return T.evaluate_root(
                world, me, pol, copy.deepcopy(xs[0]), copy.deepcopy(fs[0].aux),
                copy.deepcopy(xs[1]), copy.deepcopy(fs[1].aux), ns(fs[0]), ns(fs[1]),
                teacher=1, K=K, M=M, H=H, seed_base=7000,
                feats=(fs[0], fs[1]) if feats else None, belief=belief)

        occ = set(sim.base_at) | {t.pos for t in sim.trees.values()} | {w.pos for w in sim.workers.values()}
        v0, v1 = visible(sim, 0), visible(sim, 1)
        cells_both = [c for c in range(F.N) if not v0[c] and not v1[c] and c not in occ]
        cells_me = [c for c in range(F.N) if not v0[c] and c not in occ]
        wA = other_world(sim, [])
        wB = other_world(sim, cells_both)
        wC = other_world(sim, cells_me)
        check(f"{tag} construction: both seats' observations identical in A and B",
              all(wA.obs(p) == wB.obs(p) for p in (0, 1)), f"{len(cells_both)} hidden cells changed")
        check(f"{tag} construction: acting seat's observation identical in A and C",
              wA.obs(me) == wC.obs(me), f"{len(cells_me)} cells changed")

        sig0 = (feat_sig(fs[0]), feat_sig(fs[1]))
        rA = run(wA, "sample"); rB = run(wB, "sample"); rC = run(wC, "sample"); rA2 = run(wA, "sample")
        same = lambda r1, r2: len(r1["arms"]) == len(r2["arms"]) and all(
            np.array_equal(a, b) for a, b in zip(r1["arms"], r2["arms"]))
        dq = lambda r1, r2: float(np.abs(r1["Q_mean"] - r2["Q_mean"]).max()) if same(r1, r2) else float("inf")
        check(f"{tag} A. v2: Q unchanged when cells invisible to both seats differ",
              dq(rA, rB) == 0.0 and np.array_equal(rA["q"], rB["q"]),
              f"max|dQ|={dq(rA, rB):.3e} arms={rA['n_arms']} unknown_cells={rA['n_unknown_cells']}")
        check(f"{tag} B. v2: Q unchanged when every cell outside the acting seat's view differs",
              dq(rA, rC) == 0.0, f"max|dQ|={dq(rA, rC):.3e}")
        check(f"{tag} D. v2 deterministic", dq(rA, rA2) == 0.0, f"max|dQ|={dq(rA, rA2):.3e}")
        check(f"{tag} E. root Feats not mutated", sig0 == (feat_sig(fs[0]), feat_sig(fs[1])))
        check(f"{tag} result labels", rA["teacher_version"] == "v2" and rA["belief"] == "sample" and rA["feat_inherit"])
        oA = run(wA, "oracle"); oB = run(wB, "oracle")
        check(f"{tag} C. control: oracle clones DO depend on the hidden cells",
              dq(oA, oB) > 0.0, f"max|dQ|={dq(oA, oB):.4f} (label {oA['teacher_version']})")

        if stop == 60:
            for name, kw, exc in (("feats=None with belief='sample' refused", dict(feats=False, belief="sample"), ValueError),):
                try:
                    run(wA, kw["belief"], feats=kw["feats"]); ok = False
                except exc:
                    ok = True
                check("guard: " + name, ok)
            stale = feat_copy(fs[0]); stale.step = stale.step - 1
            try:
                T.evaluate_root(wA, me, pol, copy.deepcopy(xs[0]), copy.deepcopy(fs[0].aux), copy.deepcopy(xs[1]),
                                copy.deepcopy(fs[1].aux), ns(fs[0]), ns(fs[1]), teacher=1, K=2, M=1, H=1,
                                feats=(stale, fs[1]), belief="sample"); ok = False
            except ValueError:
                ok = True
            check("guard: Feat that did not observe the root step refused", ok)
            v1r = run(wA, None, feats=False)
            check("legacy call (no feats, default belief) is the v1 teacher and is labelled so",
                  v1r["teacher_version"] == "v1" and v1r["belief"] == "oracle" and np.isfinite(v1r["Q_mean"]).all())
            check("default belief with feats resolves to v2 sample", run(wA, None)["teacher_version"] == "v2")


    # belief_carbon unit checks on an EARLY position (most orbits still unknown;
    # by step ~60 of sampled play every mirror orbit has usually been seen)
    sim, fs, xs = root_position(pol, 21, 8)
    me = 0
    v0 = visible(sim, 0)
    players = sim.obs(me)["players"]
    unk = T.unknown_cells(fs[0], players)
    c1, n1 = T.belief_carbon(fs[0], players, sim.config, np.random.default_rng([7000, 0xC0]), "sample")
    c1b, _ = T.belief_carbon(fs[0], players, sim.config, np.random.default_rng([7000, 0xC0]), "sample")
    c2, _ = T.belief_carbon(fs[0], players, sim.config, np.random.default_rng([7001, 0xC0]), "sample")
    cm, _ = T.belief_carbon(fs[0], players, sim.config, None, "mean")
    c1, c2, cm = np.array(c1), np.array(c2), np.array(cm)
    check("belief_carbon: early position has unknown cells", n1 > 50, f"unknown={n1}")
    check("belief_carbon: known cells equal the belief", np.array_equal(c1[~unk], fs[0].est[~unk]) and n1 == int(unk.sum()))
    check("belief_carbon: visible cells exact", np.allclose(c1[v0], np.array(sim.carbon)[v0]))
    check("belief_carbon: mean mode is the belief itself", np.array_equal(cm, fs[0].est))
    orb_ok = all(len({c1[int(m)] for m in F.MIRROR[c] if unk[int(m)]}) == 1 for c in np.nonzero(unk)[0])
    check("belief_carbon: unknown mirror orbits share one sampled value", orb_ok)
    check("belief_carbon: same seed reproduces, scenario seeds differ",
          c1.tolist() == c1b and not np.array_equal(c1, c2))
    check("belief_carbon: sampled values within [0, maxCellCarbon]",
          float(c1.min()) >= 0.0 and float(c1.max()) <= float(sim.config["maxCellCarbon"]))
    nzf = float((c1[unk] > 0).mean())
    check("belief_carbon: sampled non-zero share near the generator's (29/64)", 0.25 < nzf < 0.65, f"share={nzf:.2f}")
    wA = other_world(sim, [])
    wB = other_world(sim, [c for c in range(F.N) if unk[c]])
    ns2 = lambda f: types.SimpleNamespace(seat=f.seat, own_uid=list(f.own_uid))
    ev = lambda w: T.evaluate_root(w, me, pol, copy.deepcopy(xs[0]), copy.deepcopy(fs[0].aux), copy.deepcopy(xs[1]),
                                   copy.deepcopy(fs[1].aux), ns2(fs[0]), ns2(fs[1]), teacher=1, K=K, M=M, H=H,
                                   seed_base=7000, feats=(fs[0], fs[1]), belief="sample")
    rA, rB = ev(wA), ev(wB)
    check("early position (sampled unknown cells in play): Q unchanged when all unknown cells differ",
          float(np.abs(rA["Q_mean"] - rB["Q_mean"]).max()) == 0.0 and rA["n_unknown_cells"] == n1,
          f"unknown_cells={rA['n_unknown_cells']} Q_std over scenarios max={float(rA['Q_std'].max()):.4f}")

    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
