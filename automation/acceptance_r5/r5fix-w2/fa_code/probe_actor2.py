#!/usr/bin/env python3
"""probe_actor2 -- per-channel live-vs-capture feature diff.

Teacher drives side 0 (seed 2000).  Each step, capture:
  truth: recorder row (belief, known, prev_act, seq, since_dep)
         + build-side featurize() output for that state;
  live:  the args the DistilledActor actually passes to its own
         featurize (via a recording wrapper on the module global).
Report per-channel mismatches: prev_act exact-match rate, seq /
since_dep abs diffs, and full W_x/B_x max-abs diffs.  The channel
that diverges is the deploy bug.
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src",
                                "carbon_challenge-master"))

import numpy as np  # noqa: E402
import record_trace  # noqa: E402
import distill_capture as DC  # noqa: E402
import distill_eval as DE  # noqa: E402
import distill_features as DF  # noqa: E402
import distilled_v1 as D1  # noqa: E402

SEED, SIDE = 2000, 0

calls = []
_orig = D1.featurize


def spy(me, est, known, cash, bases, workers, wmask, trees, tmask,
        step, prev_act, seq, since_dep):
    calls.append({
        "est": np.array(est), "known": np.array(known),
        "cash": np.array(cash), "bases": np.array(bases),
        "workers": np.array(workers), "wmask": np.array(wmask),
        "trees": np.array(trees), "tmask": np.array(tmask),
        "step": step, "prev_act": np.array(prev_act),
        "seq": np.array(seq), "since_dep": float(since_dep)})
    return _orig(me, est, known, cash, bases, workers, wmask, trees,
                 tmask, step, prev_act, seq, since_dep)


D1.featurize = spy


def main():
    random.seed(7919 * SEED + 104729 * SIDE + 31)
    sim = record_trace.FastCarbon(record_trace.CFG).reset(seed=SEED)
    a1 = DE.load_agent_fresh(os.path.join(HERE, "base_v53.py"))
    tmod = DC.load_module_fresh(os.path.join(HERE, "base_v53.py"))
    rec = DC.TeacherRecorder(tmod, SIDE)
    act_obj = D1.make_agent()

    pa_match = pa_tot = 0
    seq_diff, sd_diff, wx_diff, bx_diff = [], [], [], []
    for k in range(300):
        obs = [sim.obs(0), sim.obs(1)]
        t_acts = rec(obs[SIDE], record_trace.CFG)
        row = rec.rows[-1]
        _ = act_obj(obs[SIDE], record_trace.CFG)
        live = calls[-1]
        # truth-side featurize from the recorder state + live canon
        W_t, B_t = DF.featurize(
            SIDE, row["belief"], row["known"], live["cash"],
            live["bases"], live["workers"], live["wmask"],
            live["trees"], live["tmask"], k, row["prev_act"],
            row["seq"], float(row["since_dep"]))
        W_l, B_l = _orig(
            SIDE, live["est"], live["known"], live["cash"],
            live["bases"], live["workers"], live["wmask"],
            live["trees"], live["tmask"], k, live["prev_act"],
            live["seq"], live["since_dep"])
        pa_match += int((live["prev_act"] == row["prev_act"]).sum())
        pa_tot += len(row["prev_act"])
        seq_diff.append(float(np.abs(
            np.asarray(live["seq"], dtype=np.float64)
            - np.asarray(row["seq"], dtype=np.float64)).max()))
        sd_diff.append(abs(live["since_dep"]
                           - float(row["since_dep"])))
        if len(W_t) == len(W_l) and len(W_t):
            wx_diff.append(float(np.abs(W_t - W_l).max()))
        if len(B_t) == len(B_l) and len(B_t):
            bx_diff.append(float(np.abs(B_t - B_l).max()))
        acts1 = a1(obs[1], record_trace.CFG)
        sim.step(t_acts, acts1)
    print(f"prev_act match {pa_match}/{pa_tot} "
          f"= {pa_match / pa_tot:.3f}")
    print(f"seq maxdiff: mean {np.mean(seq_diff):.3f} "
          f"max {np.max(seq_diff):.3f}")
    print(f"since_dep diff: mean {np.mean(sd_diff):.2f} "
          f"max {np.max(sd_diff):.1f} "
          f"steps>0: {sum(d > 0 for d in sd_diff)}/300")
    print(f"W_x maxdiff: mean {np.mean(wx_diff):.3f} "
          f"max {np.max(wx_diff):.3f} "
          f"steps>0.01: {sum(d > 0.01 for d in wx_diff)}/300")
    print(f"B_x maxdiff: mean {np.mean(bx_diff):.3f} "
          f"max {np.max(bx_diff):.3f} "
          f"steps>0.01: {sum(d > 0.01 for d in bx_diff)}/300")


if __name__ == "__main__":
    main()
