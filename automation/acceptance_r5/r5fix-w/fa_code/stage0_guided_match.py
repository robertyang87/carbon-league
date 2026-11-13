"""Stage 0 teacher quality gate: teacher-q-guided vs raw-greedy bare net.

Standalone match driver (not bigeval): the guided side needs the live
FastCarbon sim for stage0_teacher.evaluate_root, which bigeval's agent
protocol cannot supply. The opponent side reuses the exam-stack agent
files (exp/opp_master2r.py = Master2 real public source, exp/echo2.py).

Margin/win follow eval_v53 exactly (F01/F04 terminal score):
  extinct side scores (steps_played - 300 - 1); a survivor of an extinction
  finish scores the cash it carried INTO the terminal step; a plain
  time-out finish scores final cash. margin = score[me] - score[opp].

Both arms use the SAME raw Policy (m3s2r1), guards off, so the delta
isolates the teacher: guided = argmax q arm per step; greedy = argmax
decode per step. Opponents are the exam-stack agents (Master2 real /
Echo2 search-agent CONF), unchanged.

Usage (from code/):
  python3 stage0_guided_match.py --weights <npz> --opponent master2|echo2
      --mode guided|greedy --seed0 994000 --n 200 --workers 18
      --out /home/arm/junyang/carbon/data/gate_guided_m2 --tag guided_m2
Per-game JSONL lines go to <out>/games.jsonl; a summary to <out>/summary.json.
"""
import argparse
import copy
import importlib.util
import json
import os
import sys
import time
import types

# Single-threaded numpy per worker: each worker is one CPU process.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cnet2_feat as F  # noqa: E402
from eval_v53 import CFG, EPISODE_STEPS  # noqa: E402
from fastsim import FastCarbon  # noqa: E402
from rl2_ppo_cnet2 import Cnet2Actor  # noqa: E402
import rl2_ppo_adapters as AD  # noqa: E402
import cnet2_np as NP  # noqa: E402
import stage0_teacher as T  # noqa: E402

OPP_FILES = {
    "master2": ["exp/opp_master2r.py"],
    "echo2": ["exp/echo2.py"],
}
# Opponent agent files live in the exam stack, not in this repo checkout.
OPP_BASES = [
    "/home/arm/junyang/s26stage/code",
    "/home/arm/junyang/carbon/code",
]


def load_opp_agent(name):
    for base in OPP_BASES:
        for rel in OPP_FILES[name]:
            p = os.path.join(base, rel)
            if os.path.exists(p):
                spec = importlib.util.spec_from_file_location(
                    "opp_%s_%d" % (name, os.getpid()), p)
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                return m.agent
    raise FileNotFoundError("opponent agent file for %s not found" % name)


def _ns(seat, own_uid):
    return types.SimpleNamespace(seat=seat, own_uid=list(own_uid))


def _score(stt, pre_cash, steps_played, extinct):
    score = []
    for p in range(2):
        if extinct[p]:
            score.append(steps_played - EPISODE_STEPS - 1)
        elif any(extinct):
            score.append(pre_cash[p])
        else:
            score.append(float(stt["cash"][p]))
    return score


def play_one(task):
    (weights, opp_name, mode, seed, max_steps) = task
    t0 = time.time()
    try:
        w = AD.load_npz(weights)
        pol = NP.Policy(w)
        actor = Cnet2Actor(w)
        opp_agent = load_opp_agent(opp_name)
        me = seed % 2  # bigeval seat rule: side alternates by seed parity
        opp = 1 - me
        sim = FastCarbon(CFG).reset(seed=seed)
        st_me = actor.new_game(me)
        st_opp = actor.new_game(opp)
        rng = np.random.default_rng(seed)
        steps_played = 0
        extinct = [False, False]
        pre_cash = [0.0, 0.0]
        t_ms, t_agree, t_ent, t_calls = [], 0, [], 0
        for step in range(1, max_steps + 1):
            pre = sim.get_state()
            pre_cash = [float(pre["cash"][0]), float(pre["cash"][1])]
            o_me, o_opp = sim.obs(me), sim.obs(opp)
            x_me = st_me["f"].observe(o_me)
            x_opp = st_opp["f"].observe(o_opp)
            # --- my side ---
            if mode == "guided":
                r = T.evaluate_root(
                    sim, me, pol, copy.deepcopy(x_me),
                    copy.deepcopy(st_me["f"].aux),
                    copy.deepcopy(x_opp), copy.deepcopy(st_opp["f"].aux),
                    _ns(me, st_me["f"].own_uid), _ns(opp, st_opp["f"].own_uid),
                    teacher=1, K=16, M=8, H=2,
                    seed_base=7000 + (seed % 1000) * 1000 + step)
                t_ms.append(r["wall_ms"])
                t_ent.append(r["entropy"])
                t_calls += 1
                if r["agree_greedy"]:
                    t_agree += 1
                arm = r["arms"][int(np.argmax(r["q"]))]
                a_me = st_me["f"].to_actions(np.asarray(arm, dtype=np.int64))
            else:
                y, _, _, _, _, _, _ = actor._decode(
                    x_me, st_me["f"].aux, rng, 1.0, True)
                a_me = st_me["f"].to_actions(y)
            # --- opponent side (exam-stack agent) ---
            try:
                a_opp = opp_agent(o_opp, None)
            except TypeError:
                a_opp = opp_agent(o_opp)
            a0, a1 = (a_me, a_opp) if me == 0 else (a_opp, a_me)
            sim.step(a0, a1)
            steps_played = step
            stt = sim.get_state()
            extinct = [T._extinct_state(stt, 0), T._extinct_state(stt, 1)]
            if any(extinct) or step >= 299:
                break
        stt = sim.get_state()
        score = _score(stt, pre_cash, steps_played, extinct)
        margin = round(score[me] - score[opp], 2)
        cash_margin = round(float(stt["cash"][me]) - float(stt["cash"][opp]), 2)
        rec = {"seed": seed, "side": me, "mode": mode, "opp": opp_name,
               "margin": margin, "cash_margin": cash_margin,
               "win": margin > 0, "steps": steps_played,
               "extinct": extinct, "wall_s": round(time.time() - t0, 1)}
        if mode == "guided" and t_ms:
            rec.update({
                "teacher_ms_mean": round(float(np.mean(t_ms)), 1),
                "teacher_ms_max": round(float(np.max(t_ms)), 1),
                "teacher_agree": round(t_agree / max(1, t_calls), 3),
                "teacher_entropy_mean": round(float(np.mean(t_ent)), 3),
                "teacher_calls": t_calls})
        return rec
    except Exception as e:
        import traceback
        return {"seed": seed, "mode": mode, "opp": opp_name,
                "error": repr(e)[:300],
                "tb": traceback.format_exc()[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--opponent", required=True, choices=["master2", "echo2"])
    ap.add_argument("--mode", required=True, choices=["guided", "greedy"])
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--max-steps", type=int, default=299)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = list(range(a.seed0, a.seed0 + a.n))
    tasks = [(a.weights, a.opponent, a.mode, s, a.max_steps) for s in seeds]
    rows = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    live_path = os.path.join(a.out, "games.jsonl")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(play_one, t) for t in tasks]
        with open(live_path, "a") as lf:
            for f in as_completed(futs):
                r = f.result()
                rows.append(r)
                lf.write(json.dumps(r) + "\n")
                lf.flush()
                print(json.dumps({k: r.get(k) for k in
                                  ("seed", "mode", "margin", "win", "wall_s",
                                   "teacher_ms_mean")}), flush=True)
    m = [r["margin"] for r in rows if "margin" in r]
    errs = [r for r in rows if "error" in r]
    summ = {"tag": a.tag or "%s_%s_%s" % (a.mode, a.opponent, a.seed0),
            "mode": a.mode, "opp": a.opponent, "seed0": a.seed0,
            "n": len(m), "want": a.n, "wins": sum(1 for x in m if x > 0),
            "mean": round(float(np.mean(m)), 1) if m else None,
            "se": round(float(np.std(m, ddof=1) / np.sqrt(len(m))), 1)
            if len(m) > 1 else None,
            "errors": len(errs)}
    if m and a.mode == "guided":
        tms = [r["teacher_ms_mean"] for r in rows if "teacher_ms_mean" in r]
        summ["teacher_ms_mean"] = round(float(np.mean(tms)), 1) if tms else None
        summ["teacher_agree"] = round(float(np.mean(
            [r["teacher_agree"] for r in rows if "teacher_agree" in r])), 3)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"))
    print("GATE-SUMMARY", json.dumps(summ), flush=True)
    return 0 if (len(m) == a.n and not errs) else 1


if __name__ == "__main__":
    sys.exit(main())
