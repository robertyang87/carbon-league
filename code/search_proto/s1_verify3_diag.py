#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""s1_verify3 诊断：58 根 changed=0 的归因（不产出验收证据）。

主趟（s1_verify3.py，seed 986400）采集 58 条 step 240–268 根记
录、离线 decide 成功 58、changed 0。本诊断确定性重放同一局，在
前若干触发根上抓冻结版 decide_at_root 的 Decision 全字段
（completed_rounds / abort_reason / phase_costs / total_elapsed），
区分两种解释：① SI02 deadline 在本 VM 慢基线下把整轮打掉（部署
口径基线 51ms 下不成立的 VM 伪影）；② 整轮完成但叶① H=1 下无候
选胜过 A0（晚局根的真实无改子）。

运行：python3 s1_verify3_diag.py [最多 decide 根数=10]
产出：stdout 表＋ s1_verify3_diag.json。退出码 0 = 跑完落盘。
"""

import copy
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace"))

import real_env                            # noqa: E402
from telemetry import StatsCollector       # noqa: E402
from smoke_real_s1 import fetch_pack       # noqa: E402
import s1_verify3 as v3                    # noqa: E402

MAX_DECIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT_JSON = os.path.join(_HERE, "s1_verify3_diag.json")
T0 = time.time()
WALL_CAP_S = 560.0


def main():
    pack_path = fetch_pack()
    _p, probe_agent = real_env.load_pack(pack_path)
    harness0 = real_env.RealEnvSearch(probe_agent, collector=StatsCollector())
    FastCarbon = harness0.inx.FastCarbon
    terminal_fn = harness0.inx.terminal_fn
    feat_copy = harness0.inx.mod["feat_copy"]
    key_units = harness0.inx.mod["key_units"]
    cfg = harness0.cfg
    harnesses = [real_env.RealEnvSearch(v3.make_fresh_agent(pack_path),
                                        collector=StatsCollector()),
                 real_env.RealEnvSearch(v3.make_fresh_agent(pack_path),
                                        collector=StatsCollector())]
    agents = [v3.make_fresh_agent(pack_path), v3.make_fresh_agent(pack_path)]
    sim = FastCarbon()
    sim.reset(seed=v3.SEED0)
    rows = []
    while sim.step_count < real_env.TERMINAL_STEP and len(rows) < MAX_DECIDE:
        if time.time() - T0 > WALL_CAP_S:
            print(f"诊断墙钟截止 {WALL_CAP_S:.0f}s 于 step {sim.step_count}")
            break
        step_now = sim.step_count
        obs = [sim.obs(0), sim.obs(1)]
        trig = [step_now >= 240 and bool(key_units(obs[s], cfg))
                for s in (0, 1)]
        pre = post = None
        if any(trig) and not terminal_fn(sim):
            pre = [v3.snapshot_memory(agents[0], feat_copy),
                   v3.snapshot_memory(agents[1], feat_copy)]
        actions = [None, None]
        base_cost = [0.0, 0.0]
        for s in (0, 1):
            t0 = time.perf_counter()
            actions[s] = agents[s](obs[s])
            base_cost[s] = time.perf_counter() - t0
        if pre is not None:
            post = [v3.snapshot_memory(agents[0], feat_copy),
                    v3.snapshot_memory(agents[1], feat_copy)]
            for s in (0, 1):
                if not trig[s] or len(rows) >= MAX_DECIDE:
                    continue
                rec = {"game": 0, "seed": v3.SEED0, "seat": s,
                       "step": step_now, "obs": copy.deepcopy(obs),
                       "sim_state": None, "cash_before": None,
                       "pre": pre, "post": post,
                       "a0": copy.deepcopy(actions[s]),
                       "opp_a0": copy.deepcopy(actions[1 - s]),
                       "base_cost": base_cost[s],
                       "scenario_seed": v3.SEED0 + step_now}
                try:
                    dec = v3.offline_decide(harnesses[s], rec, feat_copy)
                except Exception as exc:
                    rows.append({"step": step_now, "seat": s,
                                 "error": repr(exc)})
                    continue
                if dec is None:
                    rows.append({"step": step_now, "seat": s,
                                 "decide_none": True})
                    continue
                rows.append({
                    "step": step_now, "seat": s,
                    "base_cost_s": base_cost[s],
                    "completed_rounds": dec.completed_rounds,
                    "completed_comparisons": dec.completed_comparisons,
                    "changed": bool(dec.changed),
                    "chosen_index": dec.chosen_index,
                    "abort_reason": dec.abort_reason,
                    "total_elapsed_s": dec.total_elapsed_s,
                    "phase_costs": dec.phase_costs,
                })
                r = rows[-1]
                print(f"  step={step_now} seat={s} base={base_cost[s]:.3f}s "
                      f"rounds={r['completed_rounds']} "
                      f"comp={r['completed_comparisons']} "
                      f"changed={r['changed']} abort={r['abort_reason']} "
                      f"total={r['total_elapsed_s']:.3f}s "
                      f"phases={ {k: round(v, 3) for k, v in (r['phase_costs'] or {}).items()} }")
        sim.step(actions[0], actions[1])
        if terminal_fn(sim):
            break
    hist = {}
    for r in rows:
        key = r.get("abort_reason", "error/none")
        hist[str(key)] = hist.get(str(key), 0) + 1
    doc = {"seed": v3.SEED0, "rows": rows, "abort_hist": hist,
           "wall_s": time.time() - T0,
           "note": "诊断重放与主趟同一局（确定性）；只归因 changed=0，"
                   "不产出验收证据"}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1, default=str)
    print(f"产出：{OUT_JSON}（decide {len(rows)} 根，abort 分布 {hist}，"
          f"墙钟 {time.time() - T0:.0f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
