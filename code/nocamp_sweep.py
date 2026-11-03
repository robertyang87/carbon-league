#!/usr/bin/env python3
"""nocamp_sweep -- exhaustive check: can ANY Claude-side NN body in the
repo reproduce the OpenAI-side claim "NN vs no-camp 8 games, 6 wins,
mean +409"?

Already excluded: aux_seg2_zero (-14,414.6 vs nocamp_recon, nocamp_bench).
Candidates here (each vs nocamp_recon.py, seeds 1000-1003 x both sides
= 8 games, canonical seeding random.seed(7919*seed+104729*side+31),
policy = tested side, FastCarbon, full traces, INDEX tag nocamp_sweep):

  1. distilled_v1_hybrid.py   (NN + four shields; -15,686 vs base)
  2. distilled_v1.py          (pure distilled NN; -15,864 vs base)
  3. caseb_rl_ckpt/policy_ctx_seg2.py  (ctx seg2 champion; -9,899 vs base)
  4. casec_policy.py          (Case C allocation BC; G1 -14,023 vs base,
     instantiated per casec_report S7: weights from
     casec_ckpt/casec_bc_weights.npz, teacher opening N=40 with
     warmup, plant_lock=False -- the lock probe's identity check
     proved lock-off == S7 baseline field-for-field)

Groups 1-3 run through caseb_rl_probe._job exactly as nocamp_bench did
(open_n=0).  Group 4 runs an adapted copy of casec_g1_probe.play with
the opponent swapped to nocamp_recon.py (ledger accounting copied
verbatim).  Before the sweep, one casec validation game vs base_v53
(seed 1000, side 0, no trace) must reproduce the S7 G1 record
final [38.22, 11555.29] exactly; the outcome is recorded either way.

Pre-registered reading: a candidate reproduces +409 iff its mean
margin lies within +409 +/- 2 sample std AND wins >= 5/8.

Results: nocamp_sweep_results.json (rows + summaries + failures).
"""

import importlib.util
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

import caseb_rl_probe as probe  # noqa: E402
import record_trace  # noqa: E402

TAG = "nocamp_sweep"
OPP = "nocamp_recon.py"
INDEX = os.path.join(record_trace.TRACES_DIR, "INDEX.md")
SECTION = "## no-camp 穷尽扫荡（nocamp_sweep，2026-11-04）"
RESULTS = os.path.join(HERE, "nocamp_sweep_results.json")
CASEC_WEIGHTS = os.path.join(HERE, "casec_ckpt", "casec_bc_weights.npz")
CASEC_G1_CHECK = {"seed": 1000, "side": 0, "final": [38.22, 11555.29]}

BENCH_GROUPS = [
    ("distilled_hybrid", "distilled_v1_hybrid.py"),
    ("distilled_pure", "distilled_v1.py"),
    ("ctx_seg2", "caseb_rl_ckpt/policy_ctx_seg2.py"),
]
CASEC_GROUP = ("casec_bc", "casec_policy.py")
SEEDS = [1000, 1001, 1002, 1003]


# ---------------------------------------------------------------- casec
def _load_casec(plant_lock):
    spec = importlib.util.spec_from_file_location(
        f"casec_sweep_{os.getpid()}_{time.time_ns()}",
        os.path.join(HERE, "casec_policy.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CasecPolicy(dict(np.load(CASEC_WEIGHTS)),
                           plant_lock=plant_lock)


def casec_play(seed, side, opp_file, plant_lock=False, write=True):
    """casec_g1_probe.play with opponent swapped; S7 assembly."""
    random.seed(7919 * seed + 104729 * side + 31)
    sim = record_trace.FastCarbon(record_trace.CFG).reset(seed=seed)
    policy = _load_casec(plant_lock)
    opp = probe.load_agent_fresh(os.path.join(HERE, opp_file))
    teacher = probe.load_agent_fresh(os.path.join(HERE, "base_v53.py"))
    open_n = 40
    agents = [None, None]
    agents[side] = policy
    agents[1 - side] = opp
    frame0, st0 = record_trace.snapshot(sim)
    bases = {str(p): [[bid, pos] for bid, pos in st0["bases"][p]]
             for p in range(2)}
    frames = [frame0]
    led = [probe.new_ledger(), probe.new_ledger()]
    for k in range(300):
        pre = sim.get_state()
        acts = []
        for p in range(2):
            o = sim.obs(p)
            if p == side and k < open_n:
                tacts = teacher(o, record_trace.CFG)
                policy.warmup(o, tacts)
                acts.append(tacts)
            else:
                acts.append(agents[p](o, record_trace.CFG))
        sim.step(acts[0], acts[1])
        post = sim.get_state()
        for p in range(2):
            L = led[p]
            price = probe._plant_price(pre["trees"], p)
            rs = ps = ss = 0.0
            for tid, t in post["trees"].items():
                if t["player"] != p:
                    continue
                L["tree_income"] += float(
                    post["trees_dict"].get(tid, [None, 0])[1])
                if tid not in pre["trees"]:
                    L["plant_count"] += 1
                    ps += price
                elif pre["trees"][tid]["player"] != p:
                    L["seize_count"] += 1
                    ss += float(record_trace.CFG["seizeCost"])
            for wid, w in post["workers"].items():
                if w["player"] == p and wid not in pre["workers"]:
                    L["recruit_count"] += 1
                    rs += float(record_trace.CFG["recCollectorCost"])
            for wid, w in pre["workers"].items():
                if w["player"] == p and wid not in post["workers"]:
                    L["deaths"] += 1
                    L["death_cargo"] += float(w["carbon"])
            L["recruit_spend"] += rs
            L["plant_spend"] += ps
            L["seize_spend"] += ss
            dep_inc = (post["cash"][p] - pre["cash"][p]) + rs + ps + ss \
                - float(sum(
                    post["trees_dict"].get(tid, [None, 0])[1]
                    for tid, t in post["trees"].items()
                    if t["player"] == p))
            L["deposits"] += dep_inc
        frame, _ = record_trace.snapshot(sim)
        frame["acts"] = [
            {kk: v for kk, v in (a or {}).items() if v is not None}
            for a in acts]
        frames.append(frame)
    final = frames[-1]["cash"]
    for p in range(2):
        led[p]["stranded"] = float(sum(
            w[1] for wid, w in frames[-1]["workers"].items()
            if wid.startswith(f"player-{p}-")))
        for kk in ("deposits", "tree_income", "recruit_spend",
                   "plant_spend", "seize_spend", "death_cargo"):
            led[p][kk] = round(led[p][kk], 1)
    tname = None
    if write:
        trace = {
            "meta": {
                "format": "carbon-trace/1",
                "policy": "casec_policy",
                "opp": os.path.splitext(os.path.basename(opp_file))[0],
                "tag": TAG,
                "seed": seed,
                "policy_side": side,
                "steps": 300,
                "size": int(record_trace.CFG["size"]),
                "bases": bases,
                "final_cash": final,
                "open_n": open_n,
            },
            "frames": frames,
        }
        tname = record_trace.trace_filename("casec_policy.py", opp_file,
                                            seed, side)
        record_trace.write_trace(
            trace, os.path.join(record_trace.TRACES_DIR, tname))
    my_cash = float(final[side])
    op_cash = float(final[1 - side])
    return {
        "tag": TAG, "policy": "casec_policy",
        "opp": os.path.splitext(os.path.basename(opp_file))[0],
        "seed": seed, "side": side, "open_n": open_n, "file": tname,
        "final_cash": [float(final[0]), float(final[1])],
        "my_cash": my_cash, "opp_cash": op_cash,
        "margin": round(my_cash - op_cash, 2),
        "ledger_pol": led[side], "ledger_opp": led[1 - side],
    }


def casec_job(job):
    seed, side = job
    t0 = time.time()
    try:
        r = casec_play(seed, side, OPP, plant_lock=False, write=True)
        r["ok"] = True
        r["sec"] = round(time.time() - t0, 1)
        return r
    except Exception as e:  # noqa: BLE001
        return {"tag": TAG, "seed": seed, "side": side, "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "sec": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- shared
def summarize(rows):
    ms = [r["margin"] for r in rows]
    wins = sum(1 for m in ms if m > 0)
    losses = sum(1 for m in ms if m < 0)
    draws = len(ms) - wins - losses

    def mean(key, led="ledger_pol"):
        return round(sum(r[led][key] for r in rows) / len(rows), 1)

    std = round(statistics.stdev(ms), 1) if len(ms) > 1 else 0.0
    mmean = round(statistics.mean(ms), 1)
    return {
        "n": len(rows),
        "wins": wins, "losses": losses, "draws": draws,
        "margin_mean": mmean,
        "margin_std_sample": std,
        "margin_min": round(min(ms), 1), "margin_max": round(max(ms), 1),
        "margins": [round(r["margin"], 1) for r in
                    sorted(rows, key=lambda r: (r["seed"], r["side"]))],
        "deposits_mean": mean("deposits"),
        "tree_income_mean": mean("tree_income"),
        "death_cargo_mean": mean("death_cargo"),
        "stranded_mean": mean("stranded"),
        "opp_deposits_mean": mean("deposits", "ledger_opp"),
        "opp_tree_income_mean": mean("tree_income", "ledger_opp"),
        "opp_stranded_mean": mean("stranded", "ledger_opp"),
        "repro_409": bool(abs(mmean - 409) <= 2 * std and wins >= 5),
    }


def append_index(all_rows, verify_map):
    exists = os.path.exists(INDEX)
    text = open(INDEX).read() if exists else ""
    lines = []
    if SECTION not in text:
        lines += ["", SECTION, "",
                  "| 标签 | 我方 | 对手 | seed | 先后手（我方） | "
                  "终分（我方 : 对手） | 核对差 | 文件 |",
                  "|---|---|---|---|---|---|---|---|"]
    for r in all_rows:
        side_txt = "P0 先手" if r["side"] == 0 else "P1 后手"
        chk = verify_map.get(r["file"], "未抽查")
        lines.append(
            f"| {r['tag']} | {r['policy']} | {r['opp']} | {r['seed']} "
            f"| {side_txt} | {r['my_cash']:.2f} : {r['opp_cash']:.2f} "
            f"| {chk} | [{r['file']}]({r['file']}) |")
    with open(INDEX, "a") as f:
        f.write("\n".join(lines) + "\n")


def main():
    # ---- pre-flight: casec S7 instantiation identity check ----
    print("PREFLIGHT casec vs base seed1000 side0 (expect final "
          f"{CASEC_G1_CHECK['final']})", flush=True)
    validation = {}
    matched_lock = None
    for lock in (False, True):
        r = casec_play(CASEC_G1_CHECK["seed"], CASEC_G1_CHECK["side"],
                       "base_v53.py", plant_lock=lock, write=False)
        got = [round(x, 2) for x in r["final_cash"]]
        validation[f"plant_lock_{lock}"] = got
        print(f"  plant_lock={lock}: final={got}", flush=True)
        if got == CASEC_G1_CHECK["final"]:
            matched_lock = lock
            break
    print(f"PREFLIGHT matched_lock={matched_lock} "
          f"(sweep uses plant_lock=False per S7 spec"
          f"{'; identity confirmed' if matched_lock is False else ''})",
          flush=True)

    jobs_bench = []
    for gname, pol in BENCH_GROUPS:
        for seed in SEEDS:
            for side in (0, 1):
                jobs_bench.append((gname, pol, seed, side))
    jobs_casec = [(seed, side) for seed in SEEDS for side in (0, 1)]
    print(f"sweep jobs: bench={len(jobs_bench)} casec={len(jobs_casec)}",
          flush=True)
    results = {g: [] for g, _ in BENCH_GROUPS}
    results[CASEC_GROUP[0]] = []
    failures = {}

    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {}
        for gname, pol, seed, side in jobs_bench:
            fut = ex.submit(probe._job, (pol, OPP, seed, side, 0, TAG))
            futs[fut] = gname
        for job in jobs_casec:
            fut = ex.submit(casec_job, job)
            futs[fut] = CASEC_GROUP[0]
        for fut in as_completed(futs):
            r = fut.result()
            gname = futs[fut]
            if not r["ok"]:
                failures.setdefault(gname, []).append(r.get("error"))
                print(f"FAIL {gname} seed{r['seed']} side{r['side']}: "
                      f"{r.get('error')}", flush=True)
                continue
            r["group"] = gname
            results[gname].append(r)
            print(f"OK {gname} seed{r['seed']} side{r['side']} "
                  f"margin={r['margin']:.0f} ({r['sec']}s)", flush=True)

    # ---- rerun verification: first game of each group, in-process ----
    verify_map = {}
    bad = 0
    group_defs = [(g, p, "bench") for g, p in BENCH_GROUPS]
    group_defs.append((CASEC_GROUP[0], CASEC_GROUP[1], "casec"))
    for gname, pol, kind in group_defs:
        rows = results[gname]
        if not rows:
            continue
        r0 = sorted(rows, key=lambda r: (r["seed"], r["side"]))[0]
        if kind == "bench":
            rr = probe.play((pol, OPP, r0["seed"], r0["side"], 0, TAG))
        else:
            rr = casec_play(r0["seed"], r0["side"], OPP,
                            plant_lock=False, write=False)
        match = rr["final_cash"] == r0["final_cash"]
        verify_map[r0["file"]] = "+0.00 / +0.00" if match else "MISMATCH"
        bad += 0 if match else 1
        print(f"VERIFY {gname} {r0['file']}: "
              f"{'MATCH' if match else 'MISMATCH ' + str(rr['final_cash'])}",
              flush=True)

    all_rows = []
    summaries = {}
    for gname, _, _ in group_defs:
        rows = sorted(results[gname], key=lambda r: (r["seed"], r["side"]))
        if rows:
            summaries[gname] = summarize(rows)
            s = summaries[gname]
            print(f"SUMMARY {gname}: {s['wins']}W/{s['losses']}L "
                  f"mean={s['margin_mean']:.1f} "
                  f"std={s['margin_std_sample']:.1f} "
                  f"repro409={s['repro_409']}", flush=True)
        all_rows += rows
    append_index(all_rows, verify_map)
    with open(RESULTS, "w") as f:
        json.dump({"tag": TAG, "casec_validation": validation,
                   "casec_matched_lock": matched_lock,
                   "groups": summaries, "failures": failures,
                   "rows": all_rows},
                  f, ensure_ascii=False, indent=1)
    print(f"wrote {RESULTS}; index appended; verify_bad={bad}; "
          f"failures={failures}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
