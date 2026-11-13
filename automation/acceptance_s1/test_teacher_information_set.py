"""Reachable information-set regression for the v3 belief teacher.

Run: python automation/acceptance_s1/test_teacher_information_set.py [WEIGHTS.npz]
Uses the same optional test-only seeded U head as test_teacher_v2_twoworld.
Both seats and both non-oracle modes must be independent of the opponent's
real private observations, features and guard, not just of sim.carbon.
"""
import copy
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_teacher_v2_twoworld as A

import numpy as np
import stage0_teacher as T
import stage1_teacher_gate as G
from cnet2_search import BankruptcyGuard


class PrivateInput:
    """A poisoned input: even validation must not inspect private data."""
    def __getattr__(self, name):
        raise AssertionError(f"private opponent attribute read: {name}")

    def __getitem__(self, key):
        raise AssertionError(f"private opponent item read: {key}")


def reachable_pair(me):
    """Two supported initial maps, same legal prefix, equal own histories."""
    seed = 21
    orig = A.FastCarbon(A.CFG).reset(seed=seed)
    opp = 1 - me
    bpos = next(iter(orig.obs(me)["players"][opp][1].values()))
    target = orig._move_nbr["UP"][bpos]
    orbit = {int(c) for c in A.F.MIRROR[target]}
    maps = [list(orig.carbon), list(orig.carbon)]
    if orig.carbon[target] == 0:
        # Swap in a nonzero orbit in both worlds, preserving the map
        # generator's support count and fourfold initial symmetry.
        for c in range(A.F.N):
            co = {int(v) for v in A.F.MIRROR[c]}
            if orig.carbon[c] > 0 and not co.intersection(orbit | set(orig.base_at)):
                for j in co:
                    maps[0][j] = maps[1][j] = 0.0
                break
        else:
            raise AssertionError("no supported nonzero orbit to exchange")
    for c in orbit:
        maps[0][c], maps[1][c] = 10.0, 40.0
    out = []
    for carbon in maps:
        sim = A.FastCarbon(A.CFG).reset(seed=seed, full_carbon=carbon)
        feats = [A.F.Feat(0), A.F.Feat(1)]
        history = []
        for step in (0, 1, 2):
            obs = [sim.obs(p) for p in (0, 1)]
            history.append(copy.deepcopy(obs))
            xs = [feats[p].observe(obs[p]) for p in (0, 1)]
            if step == 0:
                sim.step(*[{bid: "RECCOLLECTOR" for bid in obs[p]["players"][p][1]} for p in (0, 1)])
            elif step == 1:
                acts = [{}, {}]
                acts[opp][next(iter(obs[opp]["players"][opp][2]))] = "UP"
                sim.step(*acts)
        out.append((sim, feats, xs, history))
    return out


def same_result(a, b):
    return (len(a["arms"]) == len(b["arms"])
            and all(np.array_equal(x, y) for x, y in zip(a["arms"], b["arms"]))
            and all(np.array_equal(a[k], b[k]) for k in ("priors", "Q_mean", "Q_std", "q"))
            and a["guard"] == b["guard"])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(A.REPO, "data", "weights", "cnet2_b128.npz")
    pol = A.load_policy(path)
    check = A.check
    for me in (0, 1):
        pair = reachable_pair(me)
        wa, wb = pair
        check(f"me={me}: whole own legal observation history equal",
              all(x[me] == y[me] for x, y in zip(wa[3], wb[3])))
        check(f"me={me}: public state and own features equal; opponent observation differs",
              wa[0].obs(me)["players"] == wb[0].obs(me)["players"]
              and all(np.array_equal(wa[2][me][k], wb[2][me][k]) for k in wa[2][me])
              and wa[3][-1][1 - me] != wb[3][-1][1 - me])

        def run(world, mode, poisoned=False):
            sim, fs, xs, _ = world
            ns = lambda f: types.SimpleNamespace(seat=f.seat, own_uid=list(f.own_uid))
            private = PrivateInput()
            own_guard = BankruptcyGuard(T.TEACHER_GUARD_CFG).snapshot()
            # It is our own known guard history; it must remain available.
            own_guard["lostw"] = 2
            return T.evaluate_root(
                sim, me, pol, copy.deepcopy(xs[me]), copy.deepcopy(fs[me].aux),
                private if poisoned else copy.deepcopy(xs[1 - me]),
                private if poisoned else copy.deepcopy(fs[1 - me].aux),
                ns(fs[me]), private if poisoned else ns(fs[1 - me]),
                teacher=1, K=8, M=8, H=2, seed_base=7000,
                feats=(fs[me], private if poisoned else fs[1 - me]), belief=mode,
                guard_snaps=(own_guard, private if poisoned else None))

        before = [(A.feat_sig(w[1][0]), A.feat_sig(w[1][1]), copy.deepcopy(w[0].get_state())) for w in pair]
        for mode in ("sample", "mean"):
            a, b = run(wa, mode), run(wb, mode)
            check(f"me={me}/{mode}: reachable private-opponent counterexample closed", same_result(a, b),
                  f"max|dQ|={np.max(np.abs(a['Q_mean'] - b['Q_mean'])):.3g}")
            poison = run(wa, mode, poisoned=True)
            check(f"me={me}/{mode}: every supplied opponent-private input ignored", same_result(a, poison))
            check(f"me={me}/{mode}: result version and approximation explicit",
                  a["teacher_version"] == f"v{T.TEACHER_VERSION}"
                  and a["opponent_model"] == "scenario-root-history")
        after = [(A.feat_sig(w[1][0]), A.feat_sig(w[1][1]), w[0].get_state()) for w in pair]
        try:
            np.testing.assert_equal(before, after)
            unchanged = True
        except AssertionError:
            unchanged = False
        check(f"me={me}: input worlds and both live Feats unmodified", unchanged)

    current = {"teacher_version": f"v{T.TEACHER_VERSION}", "belief": "sample"}
    check("gate accepts current teacher identity", G._expect_teacher(current, "sample") is current)
    try:
        G._expect_teacher({"teacher_version": "v2", "belief": "sample"}, "sample")
    except RuntimeError:
        refused = True
    else:
        refused = False
    check("gate refuses v2 result under v3 identity", refused)
    print(f"==== {sum(A.RES)}/{len(A.RES)} checks passed ====", flush=True)
    return 0 if all(A.RES) else 1


if __name__ == "__main__":
    sys.exit(main())
