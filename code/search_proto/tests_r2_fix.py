#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R2 第一组修复回归（REVIEW_SEARCH_IMPLEMENTATION_R2_20261116）。

覆盖：
  - R2-01（P1）real_env rollout 对手席 guard 历史归属：行为反例
    翻转（叶值 −117→+209）、归属矩阵（无死亡／只己方死亡／只对手
    死亡／双席既有 lostw 不同／换席）、快照源不被分支改写、
    set_opponent_agent 路径。
  - R2-02（P1）verify3 灭绝结算：调用权威逐席终局分数（灭绝席
    step_count−episodeSteps−1、存活席入步前现金）；双席、单/双
    灭绝、灭绝∩时间；Codex 控制例 −102/−46/差 −56 翻转；输出标
    原始分差非 ΔU（R2-11 本趟分工）。
  - R2-03（P2）verify3 离线 decide：_fo_snap 取 pre-call 快照，
    在线/离线对 输入（fo 历史/信念）/预测/候选/评分/选择 全等；
    附带 P3：restore 后 st['guard_base'] is guard.guard_base 别名
    重绑且两臂集合独立。
  - §5 小项（verify3）：空验证集写「无可评估样本」；镜像比较反
    射 LEFT/RIGHT；根去重键含 seed/game。

口径：真实 Echo4 包（md5 819250bf…，经 smoke_real_s1.fetch_pack
取）＋真实 FastCarbon＋真实生产 BankruptcyGuard；act_b 标签桩与
确定性策略桩只为隔离被测变量（与 Codex 探针同法），不声称自然
发生率或整局收益。原 tests_s1.py / tests_s1_fix.py 不动。

运行：python3 tests_r2_fix.py；退出码 0 = 全过。
"""

import copy
import dataclasses
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace"))

import real_env                                       # noqa: E402
import s1_verify3 as v3                               # noqa: E402
from telemetry import StatsCollector                  # noqa: E402
from smoke_real_s1 import fetch_pack                  # noqa: E402

PACK = fetch_pack()
RESULTS = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        RESULTS.append((name, f"FAIL: {exc!r}"))
        print(f"[FAIL] {name}: {exc!r}")
    else:
        RESULTS.append((name, "PASS"))
        print(f"[PASS] {name}")


def fresh_harness():
    _pack, agent = real_env.load_pack(PACK)
    return agent, real_env.RealEnvSearch(agent, collector=StatsCollector())


def craft_sim(FastCarbon, step, cash, workers, seed=986201):
    """workers: {uid: (player, pos, cargo)}，全 COLLECTOR、无树。"""
    sim = FastCarbon()
    sim.reset(seed=seed)
    q = sim.get_state()
    q.update(step=int(step), cash=[float(cash[0]), float(cash[1])],
             carbon=[0.0] * 225, trees={}, trees_dict={},
             tid_counter=[0, 0], wid_counter=[1, 1])
    q["workers"] = {uid: {"player": pl, "pos": pos, "carbon": float(cg),
                          "type": "COLLECTOR"}
                    for uid, (pl, pos, cg) in workers.items()}
    sim.set_state(q)
    return sim


def guard_state(g):
    return (g.lostw, set(g.prevw), set(g.guard_base))


# ===========================================================================
# R2-01：对手席 guard 历史归属
# ===========================================================================

def _flip_case(seat):
    """Codex 行为反例原场景：双席各 1 collector、cash 10/10、此前
    无死亡；受控 act_b 让 player1 重载 collector 从 113 向 112 移
    动（那里有 player0 轻载 collector）。旧实现对手 guard 抄己方
    历史 → lostw 凭空 1 → GUARD_NARROW 误触发逃逸、叶 −117；修复
    后对手历史干净 → 不改写、碰撞灭绝、叶 +209（seat1 视角对称
    −209，且双席 lostw 均为 0）。"""
    agent, h = fresh_harness()
    sim = craft_sim(h.inx.FastCarbon, 100, (10, 10),
                    {"player-0-worker-0": (0, 112, 0.0),
                     "player-1-worker-0": (1, 113, 100.0)})
    obs = sim.obs(seat)
    h.begin_step(obs)
    a0 = agent(obs)
    root = dataclasses.replace(h._build_root(obs, a0), acts_o={})
    orig = h.inx.mod["act_b"]

    def fixed(pol, xs, aux, teachers, forced=None):
        labs = [np.where(x["tok_mask"][:12], 0, -1).astype(np.int64)
                for x in xs]
        target = 1 if seat == 0 else 0
        labs[target][0] = h.inx.F.CANON_MOVE[4]  # global LEFT
        return np.stack(labs)

    h.inx.mod["act_b"] = fixed
    try:
        ls = h._make_rollout(root)(root.full, {}, 2, 0)
    finally:
        h.inx.mod["act_b"] = orig
    leaf = h._raw_leaf(ls)
    guards = h._last_rollout_guards
    assert ls.terminal_kind == "extinct", ls.terminal_kind
    assert "player-1-worker-0" not in ls.sim.workers, ls.sim.workers.keys()
    assert [g.lostw for g in guards] == [0, 0], [g.lostw for g in guards]
    want = 209.0 if seat == 0 else -209.0
    assert leaf == want, (seat, leaf, want)


def g_flip_seat0():
    _flip_case(0)


def g_flip_seat1():
    _flip_case(1)


def _attribution_case(seat, case):
    """归属矩阵一格：远格双 collector、act_b 全停留（无人因移动
    死亡）；死亡只经 guard 快照 prevw 与根状态之差注入（＝根步前
    刚发生的死亡在首个续弈守卫调用处的入账）。断言 rollout 后双
    席 lostw/prevw 归属正确，且两个快照源对象均未被分支改写。"""
    me, op = seat, 1 - seat
    agent, h = fresh_harness()
    my_uid, op_uid = f"player-{me}-worker-0", f"player-{op}-worker-0"
    # cash 10 < 招募成本 30：根步真实动作不会招募新工人，工人集
    # 确定（归属断言不被招募噪声污染）；死亡只经 prevw 差注入。
    sim = craft_sim(h.inx.FastCarbon, 100, (10, 10),
                    {"player-0-worker-0": (0, 49, 0.0),
                     "player-1-worker-0": (1, 200, 0.0)})
    obs = sim.obs(seat)
    h.begin_step(obs)
    a0 = agent(obs)
    root = h._build_root(obs, a0)
    # 己方快照（生产 guard 本体，rollout 只许 copy）
    h.inx.guard.lostw, h.inx.guard.prevw = 0, {my_uid}
    # 对手快照（独立对象，经登记路径进入）
    g_opp = h.inx.mod["BankruptcyGuard"](h.cfg)
    g_opp.lostw, g_opp.prevw = 0, {op_uid}
    h.set_opponent_guard(g_opp)
    want = [0, 0]
    if case == "own_death":
        h.inx.guard.prevw = {my_uid, f"player-{me}-worker-9"}
        want[0] = 1
    elif case == "opp_death":
        g_opp.prevw = {op_uid, f"player-{op}-worker-9"}
        want[1] = 1
    elif case == "preset":
        h.inx.guard.lostw, g_opp.lostw = 2, 5
        want = [2, 5]
    src_me = guard_state(h.inx.guard)
    src_opp = guard_state(g_opp)
    orig = h.inx.mod["act_b"]

    def stay(pol, xs, aux, teachers, forced=None):
        return np.stack([np.where(x["tok_mask"][:12], 0, -1).astype(np.int64)
                         for x in xs])

    h.inx.mod["act_b"] = stay
    try:
        h._make_rollout(root)(root.full, {}, 2, 0)
    finally:
        h.inx.mod["act_b"] = orig
    guards = h._last_rollout_guards
    got = [guards[0].lostw, guards[1].lostw]
    assert got == want, (seat, case, got, want)
    assert guards[0].prevw == {my_uid}, (case, guards[0].prevw)
    assert guards[1].prevw == {op_uid}, (case, guards[1].prevw)
    # 分支不得改写快照源
    assert guard_state(h.inx.guard) == src_me, "己方快照源被分支改写"
    assert guard_state(g_opp) == src_opp, "对手快照源被分支改写"
    assert guards[0] is not h.inx.guard and guards[1] is not g_opp


def g_attribution_matrix():
    for seat in (0, 1):
        for case in ("none", "own_death", "opp_death", "preset"):
            _attribution_case(seat, case)


def g_opp_agent_source():
    """decide 路径：set_opponent_agent 取对手 agent 对象自身 guard
    （lostw=7）→ rollout 对手席沿用 7、且对手 agent 本体不被改写。"""
    agent, h = fresh_harness()
    _p2, agent2 = real_env.load_pack(PACK)
    inx2 = real_env.extract_internals(agent2)
    inx2.guard.lostw = 7
    inx2.guard.prevw = {"player-1-worker-0"}
    h.set_opponent_agent(agent2)
    sim = craft_sim(h.inx.FastCarbon, 100, (100, 100),
                    {"player-0-worker-0": (0, 49, 0.0),
                     "player-1-worker-0": (1, 200, 0.0)})
    obs = sim.obs(0)
    h.begin_step(obs)
    a0 = agent(obs)
    root = h._build_root(obs, a0)
    orig = h.inx.mod["act_b"]

    def stay(pol, xs, aux, teachers, forced=None):
        return np.stack([np.where(x["tok_mask"][:12], 0, -1).astype(np.int64)
                         for x in xs])

    h.inx.mod["act_b"] = stay
    try:
        h._make_rollout(root)(root.full, {}, 2, 0)
    finally:
        h.inx.mod["act_b"] = orig
    guards = h._last_rollout_guards
    assert guards[1].lostw == 7, guards[1].lostw
    assert guard_state(inx2.guard) == (7, {"player-1-worker-0"}, set()), \
        guard_state(inx2.guard)


# ===========================================================================
# R2-02：verify3 灭绝结算（权威逐席终局分数）
# ===========================================================================

def _stub_verify_env(returns):
    """确定性策略桩（Codex 探针同法）：make_fresh_agent 按构造序
    返回 returns[i] 的固定动作；restore_agent 置空。返回还原函数。"""
    count = [0]

    def fresh(_path):
        i = count[0]
        count[0] += 1
        ret = returns[i % len(returns)]
        return lambda obs: copy.deepcopy(ret)

    saved = (v3.make_fresh_agent, v3.restore_agent)
    v3.make_fresh_agent = fresh
    v3.restore_agent = lambda *a: None
    return lambda: (setattr(v3, "make_fresh_agent", saved[0]),
                    setattr(v3, "restore_agent", saved[1]))


def _extinct_rec(seat, sim, a0, chosen):
    return {"game": 0, "seed": 986200, "step": 298, "seat": seat,
            "sim_state": sim.get_state(),
            "obs": [sim.obs(0), sim.obs(1)],
            "pre": [None, None], "a0": a0, "opp_a0": {},
            "dec": types.SimpleNamespace(chosen_index=1, action=chosen)}


def v_extinct_seat0():
    """Codex 控制例翻转：root298、己方 cash10、载货 44 的最后
    collector 安全交付（A0）或碰撞死亡（chosen）。权威：A0 −46、
    chosen 灭绝 [−2, 100] 差 −102、差之差 −56；旧输出 [10, 100]
    差 −90、−44 且误标 valid 的口径必须翻转（valid 保留，但数值
    与 basis 标注须为权威值）。"""
    _p, base = real_env.load_pack(PACK)
    h = real_env.RealEnvSearch(base)
    FC = h.inx.FastCarbon
    sim = craft_sim(FC, 298, (10, 100),
                    {"player-0-worker-0": (0, 49, 44.0),
                     "player-1-worker-0": (1, 50, 8.0)}, seed=986200)
    a0 = {"player-0-worker-0": "DOWN"}
    chosen = {"player-0-worker-0": "RIGHT"}
    rec = _extinct_rec(0, sim, a0, chosen)
    undo = _stub_verify_env([a0, {}])
    try:
        r = v3.verify_root("unused", FC, h.inx.terminal_fn,
                           h.inx.mod["feat_copy"], rec,
                           score_fn=v3.make_terminal_score_fn(h.inx))
    finally:
        undo()
    assert r["valid"] is True, r
    assert "原始分差" in r["diff_scale"] and "ΔU" in r["diff_scale"], r
    a, c = r["arms"]["a0"], r["arms"]["chosen"]
    assert a["final_basis"] == "F04_terminal_cash", a
    assert a["final_diff_seat"] == -46.0, a
    assert c["extinct"] is True and c["final_basis"] == "F01_terminal_scores", c
    assert c["final_scores"] == [-2.0, 100.0], c
    assert c["final_diff_seat"] == -102.0, c
    assert r["chosen_minus_a0_final_diff"] == -56.0, r


def v_extinct_seat1():
    """换席：root 席 1 的 collector 碰撞灭绝 → 分数 [100, −2]、
    席位差 −102；A0 到时现金差 −90；差之差与 leaf 权威交叉核对。"""
    _p, base = real_env.load_pack(PACK)
    h = real_env.RealEnvSearch(base)
    FC = h.inx.FastCarbon
    sim = craft_sim(FC, 298, (100, 10),
                    {"player-0-worker-0": (0, 49, 8.0),
                     "player-1-worker-0": (1, 50, 44.0)}, seed=986200)
    a0 = {"player-1-worker-0": "DOWN"}
    chosen = {"player-1-worker-0": "LEFT"}
    rec = _extinct_rec(1, sim, a0, chosen)
    undo = _stub_verify_env([{}, a0])
    try:
        r = v3.verify_root("unused", FC, h.inx.terminal_fn,
                           h.inx.mod["feat_copy"], rec,
                           score_fn=v3.make_terminal_score_fn(h.inx))
    finally:
        undo()
    assert r["valid"] is True, r
    a, c = r["arms"]["a0"], r["arms"]["chosen"]
    assert a["final_diff_seat"] == -90.0, a
    assert c["final_scores"] == [100.0, -2.0], c
    assert c["final_diff_seat"] == -102.0, c
    # 与 leaf 权威逐臂交叉核对（动作按席位放进 step 的对应侧）
    auth = {}
    for label, act in (("a0", a0), ("chosen", chosen)):
        s = FC(); s.set_state(rec["sim_state"]); pre = list(s.cash)
        s.step({}, act)
        auth[label] = h.inx.leaf_fn(s, 1, 0, pre)
    assert r["chosen_minus_a0_final_diff"] == auth["chosen"] - auth["a0"], \
        (r["chosen_minus_a0_final_diff"], auth)


def v_score_fn_units():
    """逐席分数函数单元：双灭绝、单灭绝（双向）、灭绝∩时间
    （step299），逐例与 leaf_fn 分差交叉核对。"""
    _p, base = real_env.load_pack(PACK)
    h = real_env.RealEnvSearch(base)
    FC = h.inx.FastCarbon
    scores = v3.make_terminal_score_fn(h.inx)

    def cross(sim, prev, want):
        got = scores(sim, prev)
        assert got == want, (got, want)
        for me in (0, 1):
            assert abs((got[me] - got[1 - me])
                       - h.inx.leaf_fn(sim, me, 0, prev)) < 1e-9, (me, got)
        assert h.inx.terminal_fn(sim)

    # 双灭绝（无工人、无树、现金低于最便宜招募）：pen 同值
    sim = craft_sim(FC, 250, (10, 10), {}, seed=986200)
    cross(sim, [10.0, 10.0], [-51.0, -51.0])
    # 只 seat0 灭绝
    sim = craft_sim(FC, 250, (10, 100),
                    {"player-1-worker-0": (1, 50, 0.0)}, seed=986200)
    cross(sim, [10.0, 100.0], [-51.0, 100.0])
    # 只 seat1 灭绝
    sim = craft_sim(FC, 250, (100, 10),
                    {"player-0-worker-0": (0, 49, 0.0)}, seed=986200)
    cross(sim, [100.0, 10.0], [100.0, -51.0])
    # 灭绝∩时间（step_count=299）：pen = 299−300−1 = −2
    sim = craft_sim(FC, 299, (10, 100),
                    {"player-1-worker-0": (1, 50, 0.0)}, seed=986200)
    cross(sim, [10.0, 100.0], [-2.0, 100.0])
    # prev_cash=None（已终局状态交接）：用 sim 自身现金
    got = scores(sim, None)
    assert got == [-2.0, 100.0], got


def v_time_terminal():
    """纯到时终局：final_basis=F04、final_scores=真实终态现金。"""
    _p, base = real_env.load_pack(PACK)
    h = real_env.RealEnvSearch(base)
    FC = h.inx.FastCarbon
    sim = craft_sim(FC, 298, (10, 100),
                    {"player-0-worker-0": (0, 49, 0.0),
                     "player-1-worker-0": (1, 50, 0.0)}, seed=986200)
    rec = _extinct_rec(0, sim, {}, {})
    undo = _stub_verify_env([{}, {}])
    try:
        r = v3.verify_root("unused", FC, h.inx.terminal_fn,
                           h.inx.mod["feat_copy"], rec,
                           score_fn=v3.make_terminal_score_fn(h.inx))
    finally:
        undo()
    assert r["valid"] is True, r
    for name in ("a0", "chosen"):
        arm = r["arms"][name]
        assert arm["extinct"] is False, arm
        assert arm["final_basis"] == "F04_terminal_cash", arm
        assert arm["final_scores"] == [10.0, 100.0], arm
    assert r["chosen_minus_a0_final_diff"] == 0.0, r


# ===========================================================================
# R2-03：离线 decide 与在线等价 + guard_base 别名
# ===========================================================================

def _capture_root(harness, sink):
    """包装 _build_root：捕获根、候选标签与逐候选叶值（在 harness
    当前 inx 口径内计算，与 decide 自身同源）。"""
    orig = harness._build_root

    def wrapped(obs, ac):
        rr = orig(obs, ac)
        sink["acts_o"] = rr.acts_o
        sink["hist"] = {k: len(v) for k, v in harness._fo_snap.hist.items()}
        sink["last_step"] = harness._fo_snap.last_step.copy()
        sink["est"] = harness._fo_snap.est.copy()
        raw = harness._gen_candidates_raw(rr, ac)
        if raw is None:
            sink["cand_labels"] = None
            sink["cand_scores"] = None
        else:
            sink["cand_labels"] = [[int(v) for v in lab] for lab in raw[1]]
            rollout = harness._make_rollout(rr)
            sink["cand_scores"] = [
                harness.leaf_adapter(rollout(rr.full, a, 1, 0))
                for a in raw[0]]
        return rr

    harness._build_root = wrapped
    return orig


def v_offline_equiv():
    """step11 根（Codex 反例同场景）：在线（begin_step→基线调用→
    decide_at_root）与离线（offline_decide 于 pre/post 快照）须在
    对手预测、fo 历史/信念、候选集、逐候选评分、最终选择上全等。
    旧离线：预测 RIGHT、历史 1 条、81 格信念不同。BUDGET 临时放
    宽为测试口径（双路同值，只验等价、不验计时）。"""
    ags, hs = [], []
    for _ in range(2):
        _p, a = real_env.load_pack(PACK)
        ags.append(a)
        hs.append(real_env.RealEnvSearch(a))
    fc = hs[0].inx.mod["feat_copy"]
    FC = hs[0].inx.FastCarbon
    sim = FC(); sim.reset(seed=986400)
    rec = None
    for step in range(12):
        obs = [sim.obs(0), sim.obs(1)]
        pre = [v3.snapshot_memory(ags[s], fc) for s in range(2)] \
            if step == 11 else None
        for s in range(2):
            hs[s].begin_step(obs[s])
        acts = [ags[s](obs[s]) for s in range(2)]
        if step == 11:
            post = [v3.snapshot_memory(ags[s], fc) for s in range(2)]
            rec = {"seat": 0, "obs": obs, "pre": pre, "post": post,
                   "a0": acts[0], "base_cost": 0.0,
                   "scenario_seed": 986400 + step}
            online_obs, online_a0 = obs[0], acts[0]
            break
        sim.step(*acts)
    for a in ags:
        a.cfg["BUDGET"] = 30.0
    on_sink, off_sink = {}, {}
    orig_on = _capture_root(hs[0], on_sink)
    try:
        dec_on = hs[0].decide_at_root(online_obs, online_a0,
                                      baseline_step_cost_s=0.0, horizon=1,
                                      scenario_seed=986400 + 11)
    finally:
        hs[0]._build_root = orig_on
    _p, a_off = real_env.load_pack(PACK)
    a_off.cfg["BUDGET"] = 30.0
    off = real_env.RealEnvSearch(a_off)
    orig_off = _capture_root(off, off_sink)
    try:
        dec_off = v3.offline_decide(off, rec, fc)
    finally:
        off._build_root = orig_off
    # 输入与预测
    assert off_sink["acts_o"] == on_sink["acts_o"], \
        (on_sink["acts_o"], off_sink["acts_o"])
    assert off_sink["hist"] == on_sink["hist"], \
        (on_sink["hist"], off_sink["hist"])
    assert np.array_equal(off_sink["last_step"], on_sink["last_step"])
    assert np.array_equal(off_sink["est"], on_sink["est"])
    # 候选与评分
    assert off_sink["cand_labels"] == on_sink["cand_labels"]
    assert off_sink["cand_scores"] == on_sink["cand_scores"], \
        (on_sink["cand_scores"], off_sink["cand_scores"])
    # 选择
    assert (dec_on is None) == (dec_off is None), (dec_on, dec_off)
    if dec_on is not None:
        assert dec_off.chosen_index == dec_on.chosen_index, (dec_on, dec_off)
        assert dec_off.action == dec_on.action, (dec_on, dec_off)
        assert dec_off.changed == dec_on.changed
        assert dec_off.completed_comparisons == dec_on.completed_comparisons


def v_restore_alias():
    """P3：restore 后 st['guard_base'] is guard.guard_base 须恢复
    为真、guard 侧变动传导到 st；两臂各自恢复后集合互独立。"""
    _p, src = real_env.load_pack(PACK)
    fc = real_env.extract_internals(src).mod["feat_copy"]
    mem = v3.snapshot_memory(src, fc)
    arms = []
    for _ in range(2):
        _p, a = real_env.load_pack(PACK)
        v3.restore_agent(a, mem, fc)
        arms.append(real_env.extract_internals(a))
    for inx in arms:
        assert inx.st["guard_base"] is inx.guard.guard_base
    arms[0].guard.guard_base.add(10)
    assert 10 in arms[0].st["guard_base"], "别名未传导"
    assert 10 not in arms[1].st["guard_base"], "两臂集合不独立"
    assert arms[0].guard.guard_base is not arms[1].guard.guard_base


# ===========================================================================
# §5 小项（verify3）
# ===========================================================================

def v_tie_items():
    # 空验证集 / 无完整臂 → 「无可评估样本」，不许写无退化迹象
    out = v3.tie_analysis([])
    assert out["seen"] is False and "无可评估样本" in out["conclusion"], out
    out = v3.tie_analysis([{"game": 0, "seed": 1, "step": 250, "seat": 0,
                            "arms": {"a0": {"complete": False},
                                     "chosen": {"complete": False}}}])
    assert "无可评估样本" in out["conclusion"], out
    # 有完整臂、无平局 → 未复现（与空集区分）
    out = v3.tie_analysis([{"game": 0, "seed": 1, "step": 250, "seat": 0,
                            "arms": {"a0": {"complete": True,
                                            "final_scores": [10.0, 20.0],
                                            "seq": []},
                                     "chosen": {"complete": True,
                                                "final_scores": [30.0, 20.0],
                                                "seq": []}}}])
    assert "未复现" in out["conclusion"], out
    # 镜像反射：LEFT<->RIGHT，UP 不变，非字符串透传
    assert v3._norm_seat1({"player-1-worker-0": "LEFT"}) == \
        {"player-0-worker-0": "RIGHT"}
    assert v3._norm_seat1({"player-1-worker-0": "RIGHT"}) == \
        {"player-0-worker-0": "LEFT"}
    assert v3._norm_seat1({"player-1-worker-0": "UP"}) == \
        {"player-0-worker-0": "UP"}
    # 去重键含 seed/game：同 step/seat 不同 seed 的两根各出 detail
    def mk(seed):
        return {"game": 0, "seed": seed, "step": 250, "seat": 0,
                "a0_action": {"player-0-worker-0": "UP"},
                "chosen_action": {"player-0-worker-0": "DOWN"},
                "arms": {"a0": {"complete": True,
                                "final_scores": [50.0, 50.0],
                                "seq": [[{"player-0-worker-0": "RIGHT"},
                                         {"player-1-worker-0": "LEFT"}]]},
                         "chosen": {"complete": True,
                                    "final_scores": [50.0, 50.0],
                                    "seq": [[{"player-0-worker-0": "RIGHT"},
                                             {"player-1-worker-0": "LEFT"}]]}}}
    out = v3.tie_analysis([mk(986400), mk(986401)])
    assert out["seen"] is True and len(out["details"]) == 2, out
    # 镜像率按反射口径：seat1 的 LEFT 与 seat0 的 RIGHT 构成镜像
    det = out["details"][0]
    assert det["within_arm_mirror_frac"] == 1.0, det


def main():
    check("R2-01 flip seat0（叶 −117→+209、对手 lostw 凭空 1→0）", g_flip_seat0)
    check("R2-01 flip seat1（叶 −209、对手 lostw 凭空 1→0）", g_flip_seat1)
    check("R2-01 归属矩阵（双席×无死亡/只己方/只对手/既有 lostw）＋快照不改写",
          g_attribution_matrix)
    check("R2-01 set_opponent_agent 取对手自身 guard", g_opp_agent_source)
    check("R2-02 灭绝 seat0 控制例翻转（−90→−102、差 −44→−56）", v_extinct_seat0)
    check("R2-02 灭绝 seat1 换席（[100,−2]、差 −102）", v_extinct_seat1)
    check("R2-02 逐席分数单元（双灭绝/单灭绝/灭绝∩时间 vs leaf）", v_score_fn_units)
    check("R2-02 纯到时终局 F04 现金口径", v_time_terminal)
    check("R2-03 离线/在线等价（预测/历史/信念/候选/评分/选择）", v_offline_equiv)
    check("R2-03 restore 后 guard_base 别名重绑＋两臂独立", v_restore_alias)
    check("§5 小项（空集文案/镜像反射/去重键）", v_tie_items)
    n_fail = sum(1 for _n, r in RESULTS if r != "PASS")
    print(f"== tests_r2_fix：{len(RESULTS) - n_fail}/{len(RESULTS)} PASS ==")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
