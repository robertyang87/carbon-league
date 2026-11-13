#!/usr/bin/env python3
"""probe_actor -- isolate the live-deploy divergence.

Play base_v53 vs base_v53 (seed 2000, protocol seeding).  For side 0,
feed every obs to BOTH a TeacherRecorder (ground truth: the exact
machinery the training data was captured with) and a fresh
DistilledActor (pure; its actions are ignored).  Per step compare:
  * belief estimate max-abs diff (actor vs recorder),
  * actor's returned acts vs the teacher's acts (exact-match rate
    per worker and per base).
If belief matches but actions disagree, the fault is forward /
sampling / action assembly; if belief diverges, it is the actor's
state maintenance.
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
import distilled_v1 as D1  # noqa: E402

SEED, SIDE = 2000, 0


def main():
    random.seed(7919 * SEED + 104729 * SIDE + 31)
    sim = record_trace.FastCarbon(record_trace.CFG).reset(seed=SEED)
    a1 = DE.load_agent_fresh(os.path.join(HERE, "base_v53.py"))
    tmod = DC.load_module_fresh(os.path.join(HERE, "base_v53.py"))
    rec = DC.TeacherRecorder(tmod, SIDE)
    act_obj = D1.make_agent()
    print("actor type:", type(act_obj))

    bel_diff = []
    w_match = w_tot = 0
    b_match = b_tot = 0
    for k in range(300):
        obs = [sim.obs(0), sim.obs(1)]
        t_acts = rec(obs[SIDE], record_trace.CFG)
        est_r = np.asarray(rec.bot.belief.estimate(), dtype=np.float32)
        my_acts = act_obj(obs[SIDE], record_trace.CFG)
        if getattr(act_obj, "belief", None) is not None:
            est_a = np.asarray(act_obj.belief.estimate(),
                               dtype=np.float32)
            bel_diff.append(float(np.abs(est_a - est_r).max()))
        for wid, a in (t_acts or {}).items():
            if wid in (my_acts or {}):
                w_tot += 1
                w_match += int(my_acts[wid] == a)
        acts1 = a1(obs[1], record_trace.CFG)
        sim.step(t_acts, acts1)
        if k % 60 == 0:
            print(f"step{k}: belief maxdiff="
                  f"{bel_diff[-1] if bel_diff else None:.3f} "
                  f"worker act match so far {w_match}/{w_tot}")
    print(f"belief maxdiff: mean={np.mean(bel_diff):.3f} "
          f"max={np.max(bel_diff):.3f} "
          f"steps>1.0: {sum(d > 1.0 for d in bel_diff)}/300")
    print(f"worker/base act exact match: {w_match}/{w_tot} "
          f"= {w_match / max(w_tot, 1):.3f}")


if __name__ == "__main__":
    main()
