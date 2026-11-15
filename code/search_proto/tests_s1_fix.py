#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 修复用例（Codex 复核 REVIEW_SEARCH_IMPLEMENTATION_20261116）。

覆盖 SI01（权威终局）/ SI02（唯一 deadline＋C_hi 整轮准入）/
SI04（续弈生产守卫）/ SI05（seat 初始化）/ SI06（有限性与失败
记录）与 §4 首条（anytime 多轮）。原 tests_s1.py 的 7 例不动、
语义不变，本文件为新增用例。

两层：
  - stub 级：纯 comparator / leaf_adapters，确定性时钟与注入。
  - 真实环境级：Echo4 包（md5 819250bf…，经 smoke_real_s1.fetch_pack
    取）＋ real_env 接线，构造根走 FastCarbon get_state/set_state
    （与 s1_battery 同法）。

运行：python3 tests_s1_fix.py；退出码 0 = 全过。
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace"))

from comparator import Candidate, Comparator          # noqa: E402
from leaf_adapters import HeuristicLeafAdapter, VNetLeafAdapter  # noqa: E402
import real_env                                       # noqa: E402
from telemetry import StatsCollector                  # noqa: E402
from smoke_real_s1 import fetch_pack                  # noqa: E402


def cand(index, action, is_a0=False):
    return Candidate(index=index, action=action, is_a0=is_a0)


def std_candidates():
    return [cand(0, "a0", is_a0=True), cand(1, "a1"), cand(2, "a2")]


# ===========================================================================
# stub 级
# ===========================================================================

def f_budget_gate():
    """SI02 C_hi 整轮准入：估算整轮成本超余额 → 不启动该轮（rollout
    零调用）、回 A0、abort=budget_gate；估算放行时整轮正常完成。"""
    calls = {"n": 0}

    def rollout(state, action, h, s):
        calls["n"] += 1
        return action

    def leaf(ls):
        return {"a0": 0.5, "a1": 0.9, "a2": 0.6}[ls]

    comp = Comparator(budget_s=0.7)
    d = comp.decide(None, std_candidates(), rollout, leaf, horizon=1,
                    scenario_seed=1, est_candidate_cost_s=0.4)  # 0.4*3=1.2>0.7
    assert d.chosen_index == 0 and d.changed is False, d
    assert d.completed_rounds == 0 and d.abort_reason == "budget_gate", d
    assert calls["n"] == 0, calls  # 整轮未启动，不是事后取消
    d2 = comp.decide(None, std_candidates(), rollout, leaf, horizon=1,
                     scenario_seed=1, est_candidate_cost_s=0.1)  # 0.3<=0.7
    assert d2.completed_rounds == 1 and d2.chosen_index == 1, d2
    assert d2.abort_reason is None, d2


def f_nonfinite_inf():
    """SI06：候选叶值 +Inf → 整轮无效、回 A0、abort=nonfinite_leaf。"""
    def rollout(state, action, h, s):
        return action

    def leaf(ls):
        return {"a0": 0.1, "a1": float("inf"), "a2": 0.3}[ls]

    comp = Comparator(budget_s=5.0)
    d = comp.decide(None, std_candidates(), rollout, leaf, horizon=1,
                    scenario_seed=1)
    assert d.chosen_index == 0 and d.changed is False, d
    assert d.completed_rounds == 0 and d.completed_comparisons == 0, d
    assert d.abort_reason == "nonfinite_leaf", d


def f_nonfinite_nan():
    """SI06：A0 叶值 NaN → 不能把无有效基线的比较记为完成。"""
    def rollout(state, action, h, s):
        return action

    def leaf(ls):
        return {"a0": float("nan"), "a1": 0.9, "a2": 0.3}[ls]

    comp = Comparator(budget_s=5.0)
    d = comp.decide(None, std_candidates(), rollout, leaf, horizon=1,
                    scenario_seed=1)
    assert d.chosen_index == 0 and d.changed is False, d
    assert d.completed_rounds == 0 and d.abort_reason == "nonfinite_leaf", d


def f_leaf_exception():
    """SI06：leaf 抛异常 → 回 A0＋失败记录（abort=exception），不逃逸。"""
    def rollout(state, action, h, s):
        return action

    def leaf(ls):
        if ls == "a2":
            raise RuntimeError("leaf boom")
        return {"a0": 0.1, "a1": 0.9}[ls]

    comp = Comparator(budget_s=5.0)
    d = comp.decide(None, std_candidates(), rollout, leaf, horizon=1,
                    scenario_seed=1)
    assert d.chosen_index == 0 and d.completed_rounds == 0, d
    assert d.abort_reason == "exception", d


def f_rollout_exception():
    """SI06：rollout 抛异常 → 回 A0＋abort=exception。"""
    def rollout(state, action, h, s):
        raise RuntimeError("rollout boom")

    def leaf(ls):
        return 0.5

    comp = Comparator(budget_s=5.0)
    d = comp.decide(None, std_candidates(), rollout, leaf, horizon=1,
                    scenario_seed=1)
    assert d.chosen_index == 0 and d.completed_rounds == 0, d
    assert d.abort_reason == "exception", d


def f_adapter_finite_vs_range():
    """SI06：适配器超域与有限性分离——原始分差 74 只记超域；
    Inf/NaN 只记非有限、不混进超域计数；vnet 的 Inf 不被 tanh 压成 1.0。"""
    ad = HeuristicLeafAdapter(lambda ls: 74.0)
    v = ad(None)
    assert v == 74.0 and ad.last_out_of_range is True, (v, ad.last_out_of_range)
    assert ad.last_nonfinite is False and ad.nonfinite_count == 0
    ad2 = HeuristicLeafAdapter(lambda ls: float("inf"))
    v2 = ad2(None)
    assert v2 == float("inf") and ad2.last_nonfinite is True
    assert ad2.nonfinite_count == 1 and ad2.out_of_range_count == 0
    assert ad2.last_out_of_range is False
    ad3 = HeuristicLeafAdapter(lambda ls: float("nan"))
    ad3(None)
    assert ad3.last_nonfinite is True and ad3.nonfinite_count == 1
    vn = VNetLeafAdapter(33, "0" * 32, backend=lambda ls: float("inf"))
    vz = vn(None)
    assert vz == float("inf") and vn.nonfinite_count == 1, vz  # 未被 tanh 掩盖
    vn2 = VNetLeafAdapter(33, "0" * 32, backend=lambda ls: 0.0)
    assert vn2(None) == 0.5


def f_anytime_h2_timeout_returns_h1():
    """anytime：H1 已完成且改子，H2 注入超时 → 返回 H1 完整轮结果。"""
    def rollout(state, action, h, s):
        if h == 2:
            time.sleep(0.30)
        return (action, h)

    vals = {("a0", 1): 0.5, ("a1", 1): 0.9, ("a2", 1): 0.6}

    def leaf(ls):
        return vals[ls]

    comp = Comparator(budget_s=0.25)
    d = comp.decide_multi(None, std_candidates(), rollout, leaf,
                          horizons=[1, 2], scenario_seed=1)
    assert d.completed_rounds == 1 and d.deepest_completed_h == 1, d
    assert d.chosen_index == 1 and d.changed is True, d
    assert d.abort_reason == "single_call_overrun", d


def f_anytime_h2_completes():
    """anytime：H1 未改（A0 最优）、H2 完成改子 → 返回 H2 结果。"""
    def rollout(state, action, h, s):
        return (action, h)

    vals = {("a0", 1): 0.9, ("a1", 1): 0.5, ("a2", 1): 0.6,
            ("a0", 2): 0.4, ("a1", 2): 0.95, ("a2", 2): 0.6}

    def leaf(ls):
        return vals[ls]

    comp = Comparator(budget_s=5.0)
    d = comp.decide_multi(None, std_candidates(), rollout, leaf,
                          horizons=[1, 2], scenario_seed=1)
    assert d.completed_rounds == 2 and d.deepest_completed_h == 2, d
    assert d.chosen_index == 1 and d.changed is True, d
    assert d.completed_comparisons == 6 and d.abort_reason is None, d


# ===========================================================================
# 真实环境级
# ===========================================================================

_PACK_PATH = None


def pack_path():
    global _PACK_PATH
    if _PACK_PATH is None:
        _PACK_PATH = fetch_pack()
    return _PACK_PATH


def fresh_harness():
    _pack, agent = real_env.load_pack(pack_path())
    return agent, real_env.RealEnvSearch(agent, collector=StatsCollector())


def state_at_step(FastCarbon, step, cash=None, seed=986200):
    """reset 标准布局，只改 step（与 cash），返回 (sim, snap)。"""
    sim = FastCarbon()
    sim.reset(seed=seed)
    snap = sim.get_state()
    snap["step"] = int(step)
    if cash is not None:
        snap["cash"] = [float(cash[0]), float(cash[1])]
    sim2 = FastCarbon()
    sim2.set_state(snap)
    assert sim2.step_count == step
    return sim2, snap


def craft(FastCarbon, step, cash, workers, carbon_cells=None, seed=986200):
    """按 spec 造状态（s1_battery 同法）：无树、碳图按 carbon_cells。

    workers: {wid: (player, pos, cargo, type)}
    """
    sim = FastCarbon()
    sim.reset(seed=seed)
    snap = sim.get_state()
    snap["step"] = int(step)
    snap["cash"] = [float(cash[0]), float(cash[1])]
    carbon = [0.0] * 225
    for pos, val in (carbon_cells or {}).items():
        carbon[int(pos)] = float(val)
    snap["carbon"] = carbon
    snap["trees"] = {}
    snap["trees_dict"] = {}
    snap["tid_counter"] = [0, 0]
    w = {}
    wc = [0, 0]
    for wid, (p, pos, cargo, kind) in workers.items():
        w[wid] = {"player": p, "pos": int(pos), "carbon": float(cargo),
                  "type": kind}
        suffix = int(str(wid).rsplit("-", 1)[1])
        wc[p] = max(wc[p], suffix + 1)
    snap["workers"] = w
    snap["wid_counter"] = wc
    sim2 = FastCarbon()
    sim2.set_state(snap)
    assert sim2.step_count == step
    return sim2, snap


def r_terminal_boundary():
    """SI01：297/298 两根×双席，H=3 rollout 推进恰止于 step299、
    终态后零策略调用（act_b 计数：298 根 0 次、297 根 1 次）。"""
    for step, want_actb in ((298, 0), (297, 1)):
        for seat in (0, 1):
            agent, harness = fresh_harness()
            FastCarbon = harness.inx.FastCarbon
            sim, _snap = state_at_step(FastCarbon, step)
            obs = sim.obs(seat)
            harness.begin_step(obs)
            a0 = agent(obs)
            assert a0, f"step{step} seat{seat} 基线空动作"
            root = harness._build_root(obs, a0)
            mod = harness.inx.mod
            orig_act_b = mod["act_b"]
            calls = {"n": 0}

            def counting_act_b(*a, **k):
                calls["n"] += 1
                return orig_act_b(*a, **k)

            mod["act_b"] = counting_act_b
            try:
                rollout = harness._make_rollout(root)
                ls = rollout(root.full, a0, 3, 986200)
            finally:
                mod["act_b"] = orig_act_b
            assert ls.sim.step_count == real_env.TERMINAL_STEP, (
                step, seat, ls.sim.step_count)
            assert ls.sim.step_count <= 299
            assert ls.left == 0 and ls.terminal_kind == "time", (
                step, seat, ls.left, ls.terminal_kind)
            assert calls["n"] == want_actb, (step, seat, calls)


def r_terminal_double_check():
    """SI01 双证（权威轴重做电池 B）：root298×双席，逐候选 H=1 叶值
    == 独立真跑一步到 step299 的终态现金差（容差 1e-6）。"""
    for seat in (0, 1):
        me, op = seat, 1 - seat
        agent, harness = fresh_harness()
        FastCarbon = harness.inx.FastCarbon
        workers = {
            f"player-{me}-worker-0": (me, 112, 40.0, "COLLECTOR"),
            f"player-{me}-worker-1": (me, 10, 0.0, "COLLECTOR"),
            f"player-{op}-worker-0": (op, 113, 2.0, "COLLECTOR"),
            f"player-{op}-worker-1": (op, 200, 0.0, "COLLECTOR"),
        }
        sim, snap = craft(FastCarbon, 298, [100, 100], workers,
                          carbon_cells={113: 100.0})
        obs = sim.obs(me)
        harness.begin_step(obs)
        a0 = agent(obs)
        assert a0, f"seat{seat} 基线空动作"
        root = harness._build_root(obs, a0)
        gen = harness._generate_candidates(root, a0)
        assert gen is not None, f"seat{seat} 无候选根（key_units 未触发）"
        candidates, _labels, _vetoed = gen
        rollout = harness._make_rollout(root)
        for c in candidates:
            ls = rollout(root.full, c.action, 1, 986200)
            v = harness.leaf_adapter(ls)
            s2 = FastCarbon()
            s2.set_state(snap)
            if me == 0:
                s2.step(c.action, root.acts_o)
            else:
                s2.step(root.acts_o, c.action)
            assert s2.step_count == 299, s2.step_count
            expected = s2.cash[me] - s2.cash[1 - me]
            assert ls.sim.step_count == 299 and ls.terminal_kind == "time", (
                seat, c.index, ls.sim.step_count, ls.terminal_kind)
            assert abs(v - expected) < 1e-6, (seat, c.index, v, expected)


def r_extinction_coincide():
    """SI01 灭绝与到时重合：root298，我方唯一 collector（载货 60）
    踏入敌轻载（2）格——同格最轻者胜，我方灭亡且 step 到 299。
    叶值须合 F01：灭绝方 pen = 299-300-1 = -2，存活方按带入步前
    现金（根现金 100）→ 叶 = -2 - 100 = -102。"""
    me, op = 0, 1
    agent, harness = fresh_harness()
    FastCarbon = harness.inx.FastCarbon
    F = harness.inx.F
    key_uid = "player-0-worker-0"
    workers = {
        key_uid: (0, 112, 60.0, "COLLECTOR"),
        "player-1-worker-0": (1, 113, 2.0, "COLLECTOR"),
        "player-1-worker-1": (1, 200, 0.0, "COLLECTOR"),
    }
    # 我方：唯一工人、无树、cash 10 < 最廉招募价 30 → 工人死即灭绝。
    # 敌方 cash 100（存活方 prev_cash 口径）。
    sim, snap = craft(FastCarbon, 298, [10, 100], workers,
                      carbon_cells={113: 100.0})
    obs = sim.obs(me)
    harness.begin_step(obs)
    a0 = agent(obs)
    assert a0
    root = harness._build_root(obs, a0)
    # 自杀候选：key 单位标签改成踏入敌格（DEST 查表，不经截断）
    slot = root.f.own_uid.index(key_uid)
    dests = F.DEST[112]
    move = None
    for m in range(5):
        if int(dests[m]) == 113:
            move = m
    assert move is not None
    y = root.y.copy()
    y[slot] = move
    suicidal = root.f.to_actions(y)
    assert suicidal.get(key_uid) != a0.get(key_uid)
    rollout = harness._make_rollout(root)
    ls = rollout(root.full, suicidal, 1, 986200)
    assert ls.sim.step_count == 299, ls.sim.step_count
    assert harness.inx.terminal_fn(ls.sim), "踏入后未达灭绝终局"
    assert ls.terminal_kind == "extinct", ls.terminal_kind
    # 敌轻载 collector 存活且吞下我方 60 货（碰撞方向 sanity；
    # 其根动作若为停采，同一步还会再采一笔，故只断下限 62）
    en = ls.sim.workers.get("player-1-worker-0")
    assert en is not None and en.carbon >= 62.0 - 1e-6, en
    assert key_uid not in ls.sim.workers
    v = harness.leaf_adapter(ls)
    assert abs(v - (-102.0)) < 1e-6, v


def r_terminal_rejection():
    """SI01 入口拒绝：step299 / step300 / 已灭绝根 → decide_at_root
    返回 None、计数与 terminal_root 失败行入账。"""
    agent, harness = fresh_harness()
    FastCarbon = harness.inx.FastCarbon
    for step in (299, 300):
        sim, _snap = state_at_step(FastCarbon, step)
        obs = sim.obs(0)
        harness.begin_step(obs)
        a0 = agent(obs)
        dec = harness.decide_at_root(obs, a0, baseline_step_cost_s=0.01)
        assert dec is None, (step, dec)
    assert harness.terminal_roots_rejected == 2, harness.terminal_roots_rejected
    # 已灭绝根（step100，对手无工人无树、cash 10 < 30）
    workers = {
        "player-0-worker-0": (0, 112, 40.0, "COLLECTOR"),
        "player-0-worker-1": (0, 10, 0.0, "COLLECTOR"),
    }
    sim, _snap = craft(FastCarbon, 100, [100, 10], workers)
    obs = sim.obs(0)
    harness.begin_step(obs)
    a0 = agent(obs)
    dec = harness.decide_at_root(obs, a0, baseline_step_cost_s=0.01)
    assert dec is None, dec
    assert harness.terminal_roots_rejected == 3
    snap = harness.collector.snapshot()
    assert snap["abort_reasons"].get("terminal_root") == 3, snap


def _key_root(seat=0, step=100):
    """R4/R6 共用：key 几何根（我方 collector 40 / 敌轻载 2 邻格）。"""
    me, op = seat, 1 - seat
    agent, harness = fresh_harness()
    FastCarbon = harness.inx.FastCarbon
    workers = {
        f"player-{me}-worker-0": (me, 112, 40.0, "COLLECTOR"),
        f"player-{me}-worker-1": (me, 10, 0.0, "COLLECTOR"),
        f"player-{op}-worker-0": (op, 113, 2.0, "COLLECTOR"),
        f"player-{op}-worker-1": (op, 200, 0.0, "COLLECTOR"),
    }
    sim, snap = craft(FastCarbon, step, [100, 100], workers,
                      carbon_cells={113: 100.0})
    obs = sim.obs(me)
    harness.begin_step(obs)
    a0 = agent(obs)
    assert a0
    return agent, harness, obs, a0, snap


def r_deadline_covers_candgen():
    """SI02 确定性时钟：候选生成注入 0.9s 慢生成＋基线 0.2s、总预算
    0.7s → 回 A0（不启动比较）、abort=deadline 有记录、遥测总耗时
    ≥0.9s 如实报、分阶段 candgen ≥0.9s。对照：不注入时整轮完成。"""
    _agent, harness, obs, a0, _snap = _key_root()
    orig = harness._gen_candidates_raw

    def slow_raw(root, a0_actions):
        time.sleep(0.9)
        return orig(root, a0_actions)

    harness._gen_candidates_raw = slow_raw
    try:
        leaf_before = harness.leaf_adapter.calls
        dec = harness.decide_at_root(obs, a0, baseline_step_cost_s=0.2)
    finally:
        harness._gen_candidates_raw = orig
    assert dec is not None and dec.chosen_index == 0, dec
    assert dec.changed is False and dec.completed_rounds == 0, dec
    assert dec.abort_reason == "deadline", dec
    assert dec.action == a0, "未回 A0"
    assert harness.leaf_adapter.calls == leaf_before, "比较器被启动了"
    assert dec.phase_costs["candgen"] >= 0.9, dec.phase_costs
    assert dec.total_elapsed_s >= 1.1, dec.total_elapsed_s
    snap = harness.collector.snapshot()
    assert snap["total_cost_accum_s"] >= 0.9, snap
    assert snap["abort_reasons"].get("deadline", 0) >= 1, snap
    assert snap["phase_cost_accum_s"].get("candgen", 0) >= 0.9, snap
    # 对照：同一根不注入慢生成 → 整轮完成。本 VM 共享 CPU 下全链
    # 实测约 1.2s（snapshot+candgen 各约 0.6s），已超部署预算 0.7s，
    # 故对照放宽 harness 的 BUDGET 到 5s 只验完成路径与记账字段；
    # 部署口径（0.7s 覆盖全链）由 S2 在考位实测，不在此冒称。
    _a2, harness2, obs2, a02, _s2 = _key_root()
    harness2.cfg["BUDGET"] = 5.0
    dec2 = harness2.decide_at_root(obs2, a02, baseline_step_cost_s=0.05)
    assert dec2 is not None and dec2.completed_rounds == 1, dec2
    assert dec2.phase_costs is not None and dec2.total_elapsed_s is not None
    phase_sum = sum(dec2.phase_costs.values())
    assert phase_sum - 1e-6 <= dec2.total_elapsed_s <= phase_sum + 0.25, (
        dec2.phase_costs, dec2.total_elapsed_s)


def r_fresh_seat1():
    """SI05：fresh seat1 根（step100），对手预测分支须全为 player-0
    动作；快照席位为 0。"""
    agent, harness = fresh_harness()
    FastCarbon = harness.inx.FastCarbon
    sim, _snap = state_at_step(FastCarbon, 100)
    obs = sim.obs(1)
    harness.begin_step(obs)
    assert harness._fo_snap.seat == 0, harness._fo_snap.seat
    a0 = agent(obs)
    assert a0
    root = harness._build_root(obs, a0)
    assert root.acts_o, "对手根动作为空"
    bad = [k for k in root.acts_o if not k.startswith("player-0-")]
    assert not bad, bad


def r_rollout_guard():
    """SI04：续弈步生产守卫——注入 act_b 让我方满载 collector 在
    H=3 两续弈步内逼近并踏入敌轻载格（最轻者胜、我方将死）；守卫
    开时危险动作被改写、工人存活且 rollout_guard_rewrites 有增；
    守卫关（对照）时工人死亡。双席守卫状态互不串扰：prevw 分别
    为两席各自 uid 集、交集为空。"""
    me, op = 0, 1
    agent, harness = fresh_harness()
    FastCarbon = harness.inx.FastCarbon
    F = harness.inx.F
    cfg = harness.cfg
    loww = int(cfg["LOWW"])
    guard_cash = float(cfg["GUARD_CASH"])
    key_uid, en_uid = "player-0-worker-0", "player-1-worker-0"
    workers = {
        # key 载货 500：rollout 的碳图是信念图 f.est（root.full 口径），
        # 敌 collector 立于信念富格、每步自动采货（实测 10→40→61→
        # 75.7），载货取 500 保证碰撞时恒重于敌、方向不被采货反转
        # （最轻者胜、重者灭亡）。
        key_uid: (0, 112, 500.0, "COLLECTOR"),
        en_uid: (1, 113, 10.0, "COLLECTOR"),
        "player-1-worker-1": (1, 200, 0.0, "COLLECTOR"),
    }
    # 补足我方工人到 nw == LOWW（窄守卫 risky 的人数条件），全为远格 planter
    for i in range(1, loww):
        workers[f"player-0-worker-{i}"] = (0, 10 + i, 0.0, "PLANTER")
    my_cash = min(20.0, guard_cash - 1.0)
    sim, snap = craft(FastCarbon, 100, [my_cash, 100], workers)
    obs = sim.obs(me)
    harness.begin_step(obs)
    a0 = agent(obs)
    assert a0
    root = harness._build_root(obs, a0)
    slot = root.f.own_uid.index(key_uid)
    en_slot = root.fo.own_uid.index(en_uid)
    n_slots = len(root.y)

    def fake_act_b(pol, xs, auxs, teachers, forced=None):
        # 根步 A0 会把 key 工人带离 112（实测 UP→97，距敌格 2），故
        # stub 从 xs 实时读双方位、每步贪心向敌格逼近：H=3 两续弈步
        # ——步1 逼近到相邻、步2 踏入敌格（敌无碳可采、载货恒 10，
        # 碰撞最轻者胜 → 我方 50 将死，除非守卫改写）。
        rows = [np.full(n_slots, -1, np.int64), np.full(n_slots, -1, np.int64)]
        mypos = int(xs[0]["tok_pos"][slot])
        epos = int(xs[1]["tok_pos"][en_slot])
        # fop 席位为 1 时其 tok_pos 在 CANON 旋转帧（实测敌 113 读作
        # 111），须转回真实帧再算 DEST/DIST；CANON 是对合。
        if int(root.f.seat) == 1:
            mypos = int(F.CANON[mypos])
        if int(root.fo.seat) == 1:
            epos = int(F.CANON[epos])
        best_m, best_d = 0, None
        for m in range(5):
            d = int(F.DIST[int(F.DEST[mypos][m]), epos])
            if best_d is None or d < best_d:
                best_d, best_m = d, m
        rows[0][slot] = best_m
        return np.stack(rows)

    mod = harness.inx.mod
    orig_act_b = mod["act_b"]
    # 窄守卫需 lostw>0：预置根守卫状态（rollout 守卫 copy 自它）
    guard = harness.inx.guard
    saved = guard.snapshot()
    guard.lostw = 1
    guard.prevw = set()
    mod["act_b"] = fake_act_b
    try:
        rollout = harness._make_rollout(root)
        rewrites_before = harness.rollout_guard_rewrites
        ls = rollout(root.full, a0, 3, 986200)
        # 守卫开：工人存活、确有改写、双席状态不串扰
        assert key_uid in ls.sim.workers, "守卫开时 key 工人仍死亡"
        assert harness.rollout_guard_rewrites > rewrites_before, "续弈步无守卫改写"
        guards = harness._last_rollout_guards
        assert guards is not None and guards[0] is not guards[1]
        assert key_uid in guards[0].prevw, guards[0].prevw
        assert en_uid in guards[1].prevw, guards[1].prevw
        assert not (guards[0].prevw & guards[1].prevw), "双席守卫状态串扰"
        # 对照：守卫关 → 同一注入下工人死亡
        cfg["GUARD"] = False
        try:
            rollout2 = harness._make_rollout(root)
            ls2 = rollout2(root.full, a0, 3, 986200)
        finally:
            cfg["GUARD"] = True
        assert key_uid not in ls2.sim.workers, "对照（守卫关）下 key 工人未死亡"
    finally:
        mod["act_b"] = orig_act_b
        guard.restore(saved)


TESTS = [
    ("F1 budget_gate (SI02)", f_budget_gate),
    ("F2 nonfinite_inf (SI06)", f_nonfinite_inf),
    ("F3 nonfinite_nan (SI06)", f_nonfinite_nan),
    ("F4 leaf_exception (SI06)", f_leaf_exception),
    ("F5 rollout_exception (SI06)", f_rollout_exception),
    ("F6 adapter_finite_vs_range (SI06)", f_adapter_finite_vs_range),
    ("F7 anytime_h2_timeout (anytime)", f_anytime_h2_timeout_returns_h1),
    ("F8 anytime_h2_completes (anytime)", f_anytime_h2_completes),
    ("R1 terminal_boundary (SI01)", r_terminal_boundary),
    ("R2 terminal_double_check (SI01)", r_terminal_double_check),
    ("R3 extinction_coincide (SI01)", r_extinction_coincide),
    ("R4 terminal_rejection (SI01)", r_terminal_rejection),
    ("R5 deadline_covers_candgen (SI02)", r_deadline_covers_candgen),
    ("R6 fresh_seat1 (SI05)", r_fresh_seat1),
    ("R7 rollout_guard (SI04)", r_rollout_guard),
]


def main():
    passed, failed = 0, 0
    for name, fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL  %s  %s" % (name, exc))
        except Exception as exc:
            failed += 1
            print("FAIL  %s  unexpected %r" % (name, exc))
        else:
            passed += 1
            print("PASS  %s" % name)
    print("SUMMARY: %d passed, %d failed, %d total"
          % (passed, failed, passed + failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
