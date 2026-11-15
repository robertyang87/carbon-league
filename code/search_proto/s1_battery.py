#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 机制电池（正式）：自然根扩量 + 精确终局正反例。

A. 自然根电池：FastCarbon 自对弈（Echo4 包基线 agent 双座驱动，包
   md5 819250bf…，SEARCH=False 原样），双席都收根。每根跑
   Comparator.decide（H=1、K≤4；budget = cfg BUDGET 0.7 − 该步基线
   步耗、下限 0.05，与 smoke_real_s1 / real_env.decide_at_root 同口径）。
   判定走 decide_instrumented()：与 decide_at_root 逐行同源的复刻
   （_build_root / _generate_candidates / _make_rollout / leaf_adapter
   全部复用 real_env 接线，仅多记每候选 leaf 值与守卫否决集），遥测
   以同一 dict 口径记入 harness.collector。
   基线动作照常下发推进对局，比较器只做同根判读（冒烟口径）。
   防失控：每局步数上限、每局根上限（强制 ≥3 局）、采集墙钟上限、
   全脚本墙钟上限；墙钟只作参考、不作部署口径（本 VM 共享 CPU）。

B. 精确终局正反例：用 FastCarbon.get_state/set_state 在真实环境里
   构造 step=299 的近终局根（无树、全图零碳，结算只由移动/碰撞/
   交付决定）。此根上 H=1 比较器的 leaf 在 left=0 处恰等于一步真
   实现金结算（WORKER_V/树值项 ∝ left 归零，CARGO_W 项只计已在
   基地格上的载货、而交付已在 step 的 phase 3 完成），故可双证：
   ① comparator 的 leaf 分差排序；② 把每个候选动作与对手根动作
   (root.acts_o) 在真图上真跑一步到 step 300 的现金差结算排序。
   正例利用碰撞规则「同格 collector 中载货最少者胜并捕获全部」
   的反直觉几何（敌轻载 collector 站在我方基地格上）：期望
   changed=True 且选中一步结算最优候选；反例期望 changed=False。
   信念图口径如实登记：fresh agent 冷启动，f.est 在不可见格为
   先验封顶值——本构造下相关机制（移动/碰撞/交付）不依赖碳图，
   collect 只发生在可见的自有格（已设零碳），故双证不受影响。

运行：python3 s1_battery.py
产出：s1_battery_roots.json（逐根明细 + 汇总 + B 逐例 + 验证段）。
退出码 0 = A 完成比较根数 > 0；其余情况 1。
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace"))

from comparator import Comparator          # noqa: E402
from telemetry import StatsCollector       # noqa: E402
import real_env                            # noqa: E402
from smoke_real_s1 import fetch_pack, pct  # noqa: E402

# ---------------- 规模与防失控参数 ----------------
SEED0 = 986300                # 诊断种子块 986200–986499 内（冒烟已用 986200 段）
ROOTS_TARGET = 48
MAX_GAMES = 5
MIN_GAMES = 3                 # 任务下限：至少 3 局（靠每局根上限保证局数）
PER_GAME_ROOT_CAP = 18
MAX_STEPS_PER_GAME = 200
COLLECT_WALL_S = 330.0        # 采集阶段墙钟上限（参考值，非部署口径）
HARD_WALL_S = 470.0           # 全脚本墙钟上限
VERIFY_MIN_STEP = 220         # 事后全程续弈验证只收晚局 changed 根
VERIFY_MAX_ROOTS = 1
OUT_JSON = os.path.join(_HERE, "s1_battery_roots.json")

T_START = time.time()


def wall():
    return time.time() - T_START


# ---------------- 插桩版 decide（与 decide_at_root 同口径） ----------------
def decide_instrumented(harness, obs, a0_actions, baseline_step_cost_s,
                        horizon=1, scenario_seed=0):
    """返回 (Decision|None, detail)。detail 含候选/leaf 值/守卫否决等明细。

    与 real_env.RealEnvSearch.decide_at_root 的唯一差别是多记明细，
    预算公式、Comparator 构造、rollout/leaf 对象、遥测 dict 均同源。
    """
    root = harness._build_root(obs, a0_actions)
    gen = harness._generate_candidates(root, a0_actions)
    if gen is None:
        return None, {"root": root, "gen_none": True}
    candidates, _labels, guard_vetoed = gen
    total = float(harness.cfg["BUDGET"])
    budget_s = total - float(baseline_step_cost_s)
    floored = False
    if budget_s < 0.05:
        budget_s = 0.05
        floored = True
    comp = Comparator(budget_s=budget_s, max_candidates=4)
    rollout0 = harness._make_rollout(root)
    idx_by_id = {id(c.action): c.index for c in candidates}
    cur = {"idx": None}

    def rollout(state, action, h, s):
        cur["idx"] = idx_by_id.get(id(action))
        return rollout0(state, action, h, s)

    leaf_values = {}
    adapter = harness.leaf_adapter

    def leaf(leaf_state):
        v = adapter(leaf_state)
        if cur["idx"] is not None:
            leaf_values[cur["idx"]] = v
        return v

    calls_before = adapter.calls
    dec = comp.decide(root.full, candidates, rollout, leaf,
                      horizon=horizon, scenario_seed=scenario_seed,
                      guard_vetoed=guard_vetoed)
    harness.collector.record_decision({
        "completed": dec.completed_rounds > 0,
        "rounds": dec.completed_rounds,
        "changed": bool(dec.changed),
        "depth": dec.deepest_completed_h if dec.completed_rounds > 0 else None,
        "leaf_calls": adapter.calls - calls_before,
        "cost_s": dec.elapsed_s,
        "abort_reason": dec.abort_reason,
    })
    detail = {
        "root": root,
        "candidates": candidates,
        "guard_vetoed": sorted(guard_vetoed),
        "leaf_values": leaf_values,
        "budget_s": budget_s,
        "budget_floored": floored,
        "leaf_calls": adapter.calls - calls_before,
    }
    return dec, detail


def root_record(game, seat, step, dec, detail, base_cost):
    return {
        "game": game, "seat": seat, "step": step,
        "n_candidates": len(detail["candidates"]),
        "guard_vetoed": detail["guard_vetoed"],
        "n_guard_vetoed": len(detail["guard_vetoed"]),
        "leaf_values": {str(k): v for k, v in detail["leaf_values"].items()},
        "chosen_index": dec.chosen_index,
        "changed": bool(dec.changed),
        "completed_rounds": dec.completed_rounds,
        "completed_comparisons": dec.completed_comparisons,
        "deepest_completed_h": dec.deepest_completed_h,
        "abort_reason": dec.abort_reason,
        "elapsed_s": dec.elapsed_s,
        "baseline_step_cost_s": base_cost,
        "budget_s": detail["budget_s"],
        "budget_floored": detail["budget_floored"],
        "leaf_calls": detail["leaf_calls"],
        "a0_action": {k: v for k, v in detail["candidates"][0].action.items()},
        "chosen_action": {k: v for k, v in dec.action.items()},
    }


# ---------------- B 部分：构造状态 ----------------
def craft_state(FastCarbon, spec):
    """按 spec 在真实 FastCarbon 里造 step=299 状态，返回 (sim, snap)。

    spec: {"cash": [c0, c1], "workers": {wid: (player, pos, cargo, type)}}
    基座布局沿 reset(seed) 的标准四基座；碳图全零、无树。
    """
    sim = FastCarbon()
    sim.reset(seed=SEED0)
    snap = sim.get_state()
    snap["step"] = 299
    snap["cash"] = list(spec["cash"])
    snap["carbon"] = [0.0] * 225
    snap["trees"] = {}
    snap["trees_dict"] = {}
    snap["tid_counter"] = [0, 0]
    workers = {}
    wc = [0, 0]
    for wid, (p, pos, cargo, kind) in spec["workers"].items():
        workers[wid] = {"player": p, "pos": int(pos), "carbon": float(cargo),
                        "type": kind}
        suffix = int(str(wid).rsplit("-", 1)[1])
        wc[p] = max(wc[p], suffix + 1)
    snap["workers"] = workers
    snap["wid_counter"] = wc
    sim2 = FastCarbon()
    sim2.set_state(snap)
    assert sim2.step_count == 299
    return sim2, snap


def run_b_case(pack_path, FastCarbon, F, name, spec, me, key_uid,
               expect_changed, design_note):
    """跑一个 B 构造例：fresh agent 冷启动 → 基线 A0 → 比较器 → 逐候选真结算。"""
    out = {"case": name, "design": design_note, "me": me, "key_uid": key_uid,
           "expect_changed": expect_changed}
    _pack, agent = real_env.load_pack(pack_path)
    harness = real_env.RealEnvSearch(agent, collector=StatsCollector())
    sim, snap = craft_state(FastCarbon, spec)
    obs = sim.obs(me)
    harness.begin_step()
    t0 = time.perf_counter()
    a0 = agent(obs)
    base_cost = time.perf_counter() - t0
    out["a0_action"] = a0
    out["baseline_step_cost_s"] = base_cost
    if not a0:
        out["error"] = "基线 agent 返回空动作（冷启动 _inner 异常兜底）"
        return out
    dec, detail = decide_instrumented(harness, obs, a0, base_cost, horizon=1,
                                      scenario_seed=SEED0 + 900)
    if dec is None:
        out["error"] = "无候选根（key_units 未触发或去重后候选 <2）"
        return out
    root = detail["root"]
    out["acts_o"] = root.acts_o
    out["guard_vetoed"] = detail["guard_vetoed"]
    out["leaf_values"] = {str(k): v for k, v in detail["leaf_values"].items()}
    out["chosen_index"] = dec.chosen_index
    out["changed"] = bool(dec.changed)
    out["abort_reason"] = dec.abort_reason
    out["key_move_a0"] = a0.get(key_uid, "STAY")
    chosen = detail["candidates"][dec.chosen_index].action
    out["key_move_chosen"] = chosen.get(key_uid, "STAY")
    # 逐候选一步真实终局结算：真图、真 step 到 300，现金差（我方口径）
    settlements = {}
    for cand in detail["candidates"]:
        s_ = FastCarbon()
        s_.set_state(snap)
        if me == 0:
            s_.step(cand.action, root.acts_o)
        else:
            s_.step(root.acts_o, cand.action)
        settlements[cand.index] = {
            "cash": [s_.cash[0], s_.cash[1]],
            "diff": s_.cash[me] - s_.cash[1 - me],
            "step_count": s_.step_count,
        }
    out["settlements"] = {str(k): v for k, v in settlements.items()}
    best_idx = max(settlements, key=lambda i: settlements[i]["diff"])
    out["settlement_best_index"] = best_idx
    out["settlement_best_unique"] = (
        sum(1 for v in settlements.values()
            if v["diff"] == settlements[best_idx]["diff"]) == 1)
    leaf = detail["leaf_values"]
    if leaf:
        leaf_best = max(leaf, key=lambda i: leaf[i])
        out["leaf_best_index"] = leaf_best
        # leaf(left=0) 应恰等于一步真结算现金差：逐候选对账（容差 1e-6）
        out["leaf_vs_settlement_max_abs_err"] = max(
            abs(leaf[i] - settlements[i]["diff"]) for i in leaf)
    out["expect_met"] = (bool(dec.changed) == expect_changed)
    out["chosen_is_settlement_best"] = (dec.chosen_index == best_idx)
    return out


# ---------------- 事后验证：changed 根全程续弈 ----------------
def verify_changed_root(pack_path, saved):
    """从 changed 根快照出发，A0 臂与 chosen 臂各用一对 fresh 基线
    agent 续弈到 300 步，比较终局现金差（根座位口径）。"""
    res = {"game": saved["game"], "seat": saved["seat"], "step": saved["step"]}
    FastCarbon = saved["FastCarbon"]
    arms = {}
    for arm_name, my_action in (("a0", saved["a0"]), ("chosen", saved["chosen"])):
        if wall() > HARD_WALL_S - 30:
            res[arm_name] = {"error": "墙钟上限前未跑"}
            continue
        _p1, va0 = real_env.load_pack(pack_path)
        _p2, va1 = real_env.load_pack(pack_path)
        sim = FastCarbon()
        sim.set_state(saved["snap"])
        if saved["seat"] == 0:
            sim.step(my_action, saved["other"])
        else:
            sim.step(saved["other"], my_action)
        agents = (va0, va1)
        while sim.step_count < 300 and wall() <= HARD_WALL_S - 10:
            b0 = agents[0](sim.obs(0))
            b1 = agents[1](sim.obs(1))
            sim.step(b0, b1)
        arms[arm_name] = {
            "final_cash": [sim.cash[0], sim.cash[1]],
            "final_diff_seat": sim.cash[saved["seat"]] - sim.cash[1 - saved["seat"]],
            "reached_step": sim.step_count,
        }
    res["arms"] = arms
    if "a0" in arms and "chosen" in arms and "final_diff_seat" in arms["a0"] \
            and "final_diff_seat" in arms["chosen"]:
        res["chosen_minus_a0_final_diff"] = (
            arms["chosen"]["final_diff_seat"] - arms["a0"]["final_diff_seat"])
    return res


# ---------------- 主流程 ----------------
def main():
    print("== vendor md5 核验 ==")
    for k, v in real_env.verify_vendor().items():
        print(f"  {k}: {v}")
    pack_path = fetch_pack()
    print(f"== Echo4 包 md5 {real_env.EXPECTED_PACK_MD5} ==")

    _p, agent0 = real_env.load_pack(pack_path)
    _p2, agent1 = real_env.load_pack(pack_path)
    agents = (agent0, agent1)
    collectors = (StatsCollector(), StatsCollector())
    harnesses = (real_env.RealEnvSearch(agent0, collector=collectors[0]),
                 real_env.RealEnvSearch(agent1, collector=collectors[1]))
    FastCarbon = harnesses[0].inx.FastCarbon
    F = harnesses[0].inx.F

    roots = []
    saved_verify = []
    games_played = 0
    steps_scanned = 0

    # ---------- A. 自然根电池 ----------
    print("== A. 自然根电池 ==")
    for g in range(MAX_GAMES):
        if len(roots) >= ROOTS_TARGET and games_played >= MIN_GAMES:
            break
        if wall() > COLLECT_WALL_S and games_played >= MIN_GAMES:
            print(f"  采集墙钟上限 {COLLECT_WALL_S}s 到，停开新局")
            break
        games_played += 1
        sim = FastCarbon()
        sim.reset(seed=SEED0 + g)
        game_roots = 0
        step = 0
        while sim.step_count < 300 and step < MAX_STEPS_PER_GAME:
            if len(roots) >= ROOTS_TARGET and games_played >= MIN_GAMES:
                break
            if game_roots >= PER_GAME_ROOT_CAP:
                break
            if wall() > COLLECT_WALL_S:
                break
            step += 1
            actions = [None, None]
            decs = [None, None]
            details = [None, None]
            for seat in (0, 1):
                obs = sim.obs(seat)
                harnesses[seat].begin_step()
                t0 = time.perf_counter()
                act = agents[seat](obs)
                base_cost = time.perf_counter() - t0
                actions[seat] = act
                steps_scanned += 1
                if game_roots < PER_GAME_ROOT_CAP and len(roots) < ROOTS_TARGET:
                    dec, detail = decide_instrumented(
                        harnesses[seat], obs, act, base_cost, horizon=1,
                        scenario_seed=SEED0 + len(roots))
                    if dec is not None:
                        decs[seat] = dec
                        details[seat] = detail
                        rec = root_record(g, seat, obs["step"], dec, detail,
                                          base_cost)
                        roots.append(rec)
                        game_roots += 1
                        print(f"  g{g} seat{seat} step={obs['step']} "
                              f"rounds={dec.completed_rounds} "
                              f"changed={dec.changed} chosen={dec.chosen_index} "
                              f"veto={len(detail['guard_vetoed'])} "
                              f"abort={dec.abort_reason} "
                              f"elapsed={dec.elapsed_s * 1000:.1f}ms "
                              f"base={base_cost * 1000:.0f}ms")
            # changed 晚局根存快照备事后验证（step 前的真状态）
            for seat in (0, 1):
                if decs[seat] is not None and decs[seat].changed \
                        and sim.step_count >= VERIFY_MIN_STEP \
                        and len(saved_verify) < VERIFY_MAX_ROOTS:
                    saved_verify.append({
                        "game": g, "seat": seat, "step": sim.step_count,
                        "snap": sim.get_state(),
                        "a0": actions[seat], "chosen": decs[seat].action,
                        "other": actions[1 - seat], "FastCarbon": FastCarbon,
                    })
            sim.step(actions[0], actions[1])
        print(f"  局 {g} 完：本局根 {game_roots}，累计 {len(roots)}，"
              f"墙钟 {wall():.0f}s")

    # ---------- B. 精确终局正反例 ----------
    print("== B. 精确终局正反例 ==")
    # 标准基座（reset 布局）：p0 = 64/160，p1 = 70/154；运行时直读校验
    probe = FastCarbon()
    probe.reset(seed=SEED0)
    base0 = probe.bases[0][0][1]
    base0b = probe.bases[0][1][1]
    base1 = probe.bases[1][0][1]
    adj = [c for c in range(225) if float(F.DIST[c, base0]) == 1.0]
    k_pos = adj[0]
    far2 = [c for c in range(225)
            if float(F.DIST[c, k_pos]) == 2.0 and float(F.DIST[c, base0]) >= 2.0
            and c not in (base0, base0b, base1, probe.bases[1][1][1])]
    e_pos_n1 = far2[0]
    extras_me = {
        "player-0-worker-1": (0, base0b, 0, "COLLECTOR"),
        "player-0-worker-2": (0, base0b, 0, "COLLECTOR"),
    }
    extra_en = {"player-1-worker-1": (1, base1, 0, "COLLECTOR")}
    b_results = []

    # 正例 P2：敌轻载 collector 站我方基地格，捕获它反而被它「捕获后交付」
    spec_p2 = {"cash": [100, 100], "workers": {
        "player-0-worker-0": (0, k_pos, 44, "COLLECTOR"),
        "player-1-worker-0": (1, base0, 8, "COLLECTOR"),
        **extras_me, **extra_en}}
    try:
        r = run_b_case(pack_path, FastCarbon, F, "P2_capture_deposit", spec_p2,
                       me=0, key_uid="player-0-worker-0", expect_changed=True,
                       design_note="敌 collector(8) 站我方基地格 64，我方 collector(44) 邻格；"
                                   "我方踏入基地格→碰撞最少载货者(敌8)胜并捕获44→敌载货52 仍在我方"
                                   "基地格→phase3 全额交付我方(+52)；不踏入则敌仅交付 8、我方 44 滞留。"
                                   "期望 changed=True 且选中踏入候选。")
        b_results.append(r)
        print(f"  P2: changed={r.get('changed')} chosen={r.get('chosen_index')} "
              f"settle_best={r.get('settlement_best_index')} err={r.get('error')}")
    except Exception as exc:  # 构造/接线异常如实登记，不冒充
        b_results.append({"case": "P2_capture_deposit", "exception": repr(exc)})
        print(f"  P2 异常：{exc!r}")

    # 正例 P1（P2 不达时期望的备选几何）：敌重载站我方基地格，我方轻载踏入捕获后交付
    spec_p1 = {"cash": [100, 100], "workers": {
        "player-0-worker-0": (0, k_pos, 12, "COLLECTOR"),
        "player-1-worker-0": (1, base0, 44, "COLLECTOR"),
        **extras_me, **extra_en}}
    try:
        r = run_b_case(pack_path, FastCarbon, F, "P1_light_capture", spec_p1,
                       me=0, key_uid="player-0-worker-0", expect_changed=True,
                       design_note="敌 collector(44) 站我方基地格，我方 collector(12) 邻格；"
                                   "踏入→我方载货最少胜、捕获44→载货56 在基地格交付(+56)；"
                                   "不踏入则敌自行交付 44(+44)。期望 changed=True 且选中踏入候选。")
        b_results.append(r)
        print(f"  P1: changed={r.get('changed')} chosen={r.get('chosen_index')} "
              f"settle_best={r.get('settlement_best_index')} err={r.get('error')}")
    except Exception as exc:
        b_results.append({"case": "P1_light_capture", "exception": repr(exc)})
        print(f"  P1 异常：{exc!r}")

    # 反例 N1：A0 交付即唯一最优
    spec_n1 = {"cash": [100, 100], "workers": {
        "player-0-worker-0": (0, k_pos, 44, "COLLECTOR"),
        "player-1-worker-0": (1, e_pos_n1, 8, "COLLECTOR"),
        **extras_me, **extra_en}}
    try:
        r = run_b_case(pack_path, FastCarbon, F, "N1_deposit_optimal", spec_n1,
                       me=0, key_uid="player-0-worker-0", expect_changed=False,
                       design_note="我方 collector(44) 邻基地格、基地格空；敌 collector(8) 距我方"
                                   "collector 2 格触发 key_units 但够不到基地/我方单位。踏入基地"
                                   "交付 +44 为唯一最优，期望 A0 即此动作、changed=False。")
        b_results.append(r)
        print(f"  N1: changed={r.get('changed')} chosen={r.get('chosen_index')} "
              f"settle_best={r.get('settlement_best_index')} err={r.get('error')}")
    except Exception as exc:
        b_results.append({"case": "N1_deposit_optimal", "exception": repr(exc)})
        print(f"  N1 异常：{exc!r}")

    # ---------- 事后验证（时间允许才跑） ----------
    verifications = []
    for saved in saved_verify:
        if wall() > HARD_WALL_S - 60:
            print("  墙钟不足，事后验证跳过")
            break
        print(f"== 事后验证：g{saved['game']} seat{saved['seat']} "
              f"step={saved['step']} changed 根全程续弈 ==")
        try:
            verifications.append(verify_changed_root(pack_path, saved))
        except Exception as exc:
            verifications.append({"game": saved["game"], "seat": saved["seat"],
                                  "step": saved["step"], "exception": repr(exc)})

    # ---------- 汇总 ----------
    elapsed = [r["elapsed_s"] for r in roots]
    completed = [r for r in roots if r["completed_rounds"] > 0]
    aborted = [r for r in roots if r["completed_rounds"] == 0]
    abort_dist = {}
    for r in aborted:
        abort_dist[r["abort_reason"]] = abort_dist.get(r["abort_reason"], 0) + 1
    depth_hist = {}
    for r in completed:
        d = str(r["deepest_completed_h"])
        depth_hist[d] = depth_hist.get(d, 0) + 1
    summary = {
        "games_played": games_played,
        "steps_scanned": steps_scanned,
        "total_roots": len(roots),
        "completed_roots": len(completed),
        "aborted_roots": len(aborted),
        "abort_reasons": abort_dist,
        "changed_count": sum(1 for r in roots if r["changed"]),
        "total_guard_vetoed_candidates": sum(r["n_guard_vetoed"] for r in roots),
        "guard_exceptions_total": sum(h._guard_exceptions for h in harnesses),
        "total_leaf_calls": sum(r["leaf_calls"] for r in roots),
        "elapsed_s_dist": {
            "mean": float(np.mean(elapsed)) if elapsed else None,
            "p50": pct(elapsed, 50) if elapsed else None,
            "p95": pct(elapsed, 95) if elapsed else None,
            "max": float(np.max(elapsed)) if elapsed else None,
        },
        "depth_hist": depth_hist,
        "collector_snapshots": [c.snapshot() for c in collectors],
        "wall_s_total": wall(),
        "notes": [
            "scenario_seed 未消费：sweep 路径无种子轴，J=1 确定性（real_env 登记口径）",
            "叶①为原始分差尺度 affine 恒等直通；leaf_values 即原始分差",
            "墙钟为本 VM 共享 CPU 参考值，不作部署口径",
            f"采集参数：target={ROOTS_TARGET} max_games={MAX_GAMES} "
            f"per_game_cap={PER_GAME_ROOT_CAP} max_steps/game={MAX_STEPS_PER_GAME} "
            f"collect_wall={COLLECT_WALL_S}s hard_wall={HARD_WALL_S}s seed0={SEED0}",
        ],
    }
    doc = {"summary": summary, "roots": roots, "part_b": b_results,
           "verification": verifications}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print("== 电池汇总 ==")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"逐根明细：{OUT_JSON}")
    ok = len(completed) > 0
    print("BATTERY:", "PASS" if ok else "FAIL（无完成比较的根）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
