"""Stage 0 verification 2: root-candidate teacher single-step cost on DGX.

Generates real positions from 2 bare-net self-play games (m3s2r1 vs itself,
temp 1.0), then runs stage0_teacher.evaluate_root (K=16, M=8, H=2) on 12
positions and reports ms per root step + per-game extrapolation.

Run from code/ with:  env -u CNET2_TEMP_KIND CNET2_TEACHER=1 python3 stage0_measure_teacher.py <npz> [steps_per_game]
Prints one JSON line per position + a TEACHER_COST summary line.
"""
import copy
import os
import sys
import time
import json
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cnet2_feat as F  # noqa: E402
from eval_v53 import CFG  # noqa: E402
from fastsim import FastCarbon  # noqa: E402
from rl2_ppo_cnet2 import Cnet2Actor  # noqa: E402
import rl2_ppo_adapters as AD  # noqa: E402
import cnet2_np as NP  # noqa: E402
import stage0_teacher as T  # noqa: E402
from cnet2_search import feat_copy  # noqa: E402

SAMPLE_STEPS = (40, 80, 120, 160, 200, 240)


def _copy_x(x):
    return {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else copy.deepcopy(v))
            for k, v in x.items()}


def _ns(seat, own_uid):
    return types.SimpleNamespace(seat=seat, own_uid=list(own_uid))


def main():
    npz = sys.argv[1]
    steps_per_game = float(sys.argv[2]) if len(sys.argv) > 2 else None
    weights = AD.load_npz(npz)
    pol = NP.Policy(weights)
    actor = Cnet2Actor(weights)
    positions = []
    for gi, seed in enumerate((995100, 995101)):
        sim = FastCarbon(CFG).reset(seed=seed)
        st0 = actor.new_game(0)
        st1 = actor.new_game(1)
        rng = np.random.default_rng(seed)
        for step in range(1, 300):
            o0, o1 = sim.obs(0), sim.obs(1)
            x0 = st0["f"].observe(o0)
            x1 = st1["f"].observe(o1)
            y0, _, _, _, _, _, _ = actor._decode(x0, st0["f"].aux, rng, 1.0, False)
            y1, _, _, _, _, _, _ = actor._decode(x1, st1["f"].aux, rng, 1.0, False)
            a0 = st0["f"].to_actions(y0)
            a1 = st1["f"].to_actions(y1)
            if step in SAMPLE_STEPS:
                me = step % 2
                xm = _copy_x(x0 if me == 0 else x1)
                xo = _copy_x(x1 if me == 0 else x0)
                auxm = copy.deepcopy(st0["f"].aux if me == 0 else st1["f"].aux)
                auxo = copy.deepcopy(st1["f"].aux if me == 0 else st0["f"].aux)
                uid_m = list(st0["f"].own_uid if me == 0 else st1["f"].own_uid)
                uid_o = list(st1["f"].own_uid if me == 0 else st0["f"].own_uid)
                positions.append({"full": sim.full_observation(0), "step": step, "me": me,
                                  "x_me": xm, "aux_me": auxm, "x_opp": xo, "aux_opp": auxo,
                                  "uid_me": uid_m, "uid_opp": uid_o,
                                  # teacher v2: private copies of both live Feats at the root
                                  "f_me": feat_copy(st0["f"] if me == 0 else st1["f"]),
                                  "f_opp": feat_copy(st1["f"] if me == 0 else st0["f"])})
            sim.step(a0, a1)
            stt = sim.get_state()
            if T._extinct_state(stt, 0) or T._extinct_state(stt, 1):
                break
    print(json.dumps({"positions": len(positions),
                      "steps": [p["step"] for p in positions]}), flush=True)
    ms_list = []
    for i, p in enumerate(positions):
        sim = FastCarbon(CFG)
        sim.load_observation(p["full"])
        r = T.evaluate_root(sim, p["me"], pol, p["x_me"], p["aux_me"], p["x_opp"], p["aux_opp"],
                            _ns(p["me"], p["uid_me"]), _ns(1 - p["me"], p["uid_opp"]),
                            teacher=1, K=16, M=8, H=2, seed_base=7000,
                            feats=(p["f_me"], p["f_opp"]), belief="sample")
        ms_list.append(r["wall_ms"])
        print(json.dumps({"pos": i, "step": p["step"], "me": p["me"],
                          "ms": r["wall_ms"], "n_arms": r["n_arms"],
                          "entropy": round(r["entropy"], 3),
                          "agree_greedy": r["agree_greedy"],
                          "greedy_q_rank": r["greedy_q_rank"],
                          "q_max": round(float(r["q"].max()), 3),
                          "q_greedy": round(float(r["q"][0]), 3),
                          "Q_mean_min": round(float(r["Q_mean"].min()), 3),
                          "Q_mean_max": round(float(r["Q_mean"].max()), 3),
                          "n_act_calls": r["n_act_calls"]}), flush=True)
    ms = np.array(ms_list, dtype=np.float64)
    summary = {"positions": len(ms_list), "ms_mean": round(float(ms.mean()), 1),
               "ms_p50": round(float(np.median(ms)), 1),
               "ms_p95": round(float(np.percentile(ms, 95)), 1),
               "ms_max": round(float(ms.max()), 1),
               "act_calls_per_step": None}
    if steps_per_game:
        summary["steps_per_game"] = steps_per_game
        summary["teacher_s_per_game"] = round(float(ms.mean()) * steps_per_game / 1000.0, 1)
        summary["teacher_min_per_game"] = round(float(ms.mean()) * steps_per_game / 60000.0, 2)
    print("TEACHER_COST", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
