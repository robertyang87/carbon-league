#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R2 修复回归（REVIEW_SEARCH_IMPLEMENTATION_R2_20261116 第二组）。

覆盖 R2-05（正式闸加固）/ R2-04 残余（deadline 锚点＋C_hi 分桶）/
R2-06（smoke 取包）/ R2-07（pack_bridge 加载序与绑定）。原
tests_s1.py / tests_s1_fix.py 不动、本文件为新增回归。

运行：python3 tests_s1_fix_r2.py；退出码 0 = 全过。
"""

import copy
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace"))

import s1_battery2 as b2                       # noqa: E402
import pack_bridge as pb                       # noqa: E402
import real_env                                # noqa: E402
from comparator import Candidate, Comparator   # noqa: E402
from leaf_adapters import HeuristicLeafAdapter  # noqa: E402
from telemetry import StatsCollector           # noqa: E402


def _doc_meta(doc):
    return {"settlement_rule": doc.get("settlement_rule"),
            "source_md5s": doc.get("source_md5s"),
            "pack_md5": (doc.get("pack") or {}).get("md5"),
            "conf_snapshot": doc.get("conf_snapshot"),
            "fixture_identity": doc.get("fixture_identity")}


def _formal_doc():
    with open(os.path.join(_HERE, "s1_battery2_formal_evidence.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def _by_id(assertions):
    return {a["id"]: a for a in assertions}


# ===========================================================================
# R2-05 正式闸
# ===========================================================================

def g1_formal_artifact_recompute():
    """已提交正式工件（b564b154）按加固闸重算：A8 必须显形两例 B 腿
    single_call_overrun（旧闸误报 PASS）；带中止的反例不再算达
    （A3 随之 FAIL）；其余实质项（A4/A5/A6/A7/A9）仍 PASS。"""
    doc = _formal_doc()
    aa = _by_id(b2.evaluate_formal(doc["summary"], doc["roots"],
                                   doc["part_b"], doc["verification"],
                                   _doc_meta(doc)))
    assert aa["A8"]["status"] == "FAIL", aa["A8"]
    assert "B_capture_step297_seat1" in aa["A8"]["detail"], aa["A8"]
    assert "B_deposit_step297_seat1" in aa["A8"]["detail"], aa["A8"]
    assert aa["A2"]["status"] == "FAIL", aa["A2"]
    assert aa["A3"]["status"] == "FAIL", aa["A3"]  # 中止负例不得冒充达标
    for aid in ("A1", "A4", "A5", "A6", "A7", "A9"):
        assert aa[aid]["status"] == "PASS", (aid, aa[aid])
    assert b2.verdict_of(list(aa.values())) == "FAIL"


def g2_synthetic_positive_control_keeps_aborts():
    """合成对抗：正例 expect_met 全改 True、保留 B 腿中止 → 旧闸整体
    PASS，加固后 A8 FAIL、整体不得 PASS（Codex 探针同构）。"""
    doc = _formal_doc()
    for c in doc["part_b"]:
        if c.get("expect_changed") is True:
            c["expect_met"] = True
    aa = _by_id(b2.evaluate_formal(doc["summary"], doc["roots"],
                                   doc["part_b"], doc["verification"],
                                   _doc_meta(doc)))
    assert aa["A8"]["status"] == "FAIL", aa["A8"]
    assert b2.verdict_of(list(aa.values())) == "FAIL"


def g3_synthetic_nan_errors():
    """合成对抗：B 腿误差全改 NaN（中止清除、expect 全 True）→
    旧闸 A5 PASS，加固后 A5 FAIL（finite 且 0<=v<=tol）。"""
    doc = _formal_doc()
    for c in doc["part_b"]:
        c["abort_reason"] = None
        if c.get("expect_changed") is True:
            c["expect_met"] = True
        if "leaf_vs_settlement_max_abs_err" in c:
            c["leaf_vs_settlement_max_abs_err"] = float("nan")
    aa = _by_id(b2.evaluate_formal(doc["summary"], doc["roots"],
                                   doc["part_b"], doc["verification"],
                                   _doc_meta(doc)))
    assert aa["A5"]["status"] == "FAIL", aa["A5"]
    assert b2.verdict_of(list(aa.values())) == "FAIL"


def g4_synthetic_missing_canonical_one_root():
    """合成对抗：只剩 2 个名单外用例＋1 根、summary 谎报 48/3 →
    A7 逐根重算 FAIL、A2–A5 正典缺失 FAIL，整体不得 PASS。"""
    doc = _formal_doc()
    doc["part_b"] = [
        {"case": "SYNTHETIC_POS", "expect_changed": True,
         "expect_met": True, "chosen_is_settlement_best": True,
         "leaf_vs_settlement_max_abs_err": 0.0},
        {"case": "SYNTHETIC_NEG", "expect_changed": False,
         "expect_met": True},
    ]
    doc["roots"] = doc["roots"][:1]
    aa = _by_id(b2.evaluate_formal(doc["summary"], doc["roots"],
                                   doc["part_b"], doc["verification"],
                                   _doc_meta(doc)))
    assert aa["A7"]["status"] == "FAIL", aa["A7"]
    for aid in ("A2", "A3", "A4", "A5"):
        assert aa[aid]["status"] == "FAIL", (aid, aa[aid])
    assert b2.verdict_of(list(aa.values())) == "FAIL"


def g5_fixture_identity_params():
    """P3：fixture 身份纳入全部构造参数——改 step/seat 身份必变；
    expect_changed 的 None 与 False 必须可区分。"""
    c0 = {"case": "B_capture_step298_seat0", "expect_changed": True,
          "step": 298, "me": 0, "spec": {"cash": [100, 100], "workers": {}}}
    c1 = copy.deepcopy(c0)
    c1["step"] = 100
    c1["me"] = 1
    h0 = b2.fixture_identity([c0], 986300, {})["part_b_cases"][0]
    h1 = b2.fixture_identity([c1], 986300, {})["part_b_cases"][0]
    assert h0["identity_sha256"] != h1["identity_sha256"], (h0, h1)
    cn = {"case": "REJ_x", "expect_changed": None, "step": 299, "me": 0}
    cf = {"case": "REJ_x", "expect_changed": False, "step": 299, "me": 0}
    hn = b2.fixture_identity([cn], 986300, {})["part_b_cases"][0]
    hf = b2.fixture_identity([cf], 986300, {})["part_b_cases"][0]
    assert hn["identity_sha256"] != hf["identity_sha256"]
    assert hn["expect_changed"] is None and hf["expect_changed"] is False


# ===========================================================================
# R2-04 残余：deadline 锚点（确定性时钟，Codex 探针同构）
# ===========================================================================

class _Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def tick(self, dt):
        self.t += dt


def _fake_harness(clock, prep=0.0, begin=0.0):
    h = real_env.RealEnvSearch.__new__(real_env.RealEnvSearch)
    h.cfg = {"BUDGET": 0.7}
    h.notes = []
    h.collector = StatsCollector()
    h.terminal_roots_rejected = 0
    h._cand_cost_sum = 0
    h._cand_cost_n = 0
    h.leaf_adapter = HeuristicLeafAdapter(
        lambda a: 0.1 if a == "a0" else 0.9)
    h._fo_snap = None
    phase_calls = []

    class Feat:
        def __init__(self, seat):
            self.seat = seat

    def copy_feat(fo):
        clock.tick(begin)
        phase_calls.append("begin_snapshot")
        return fo

    h.inx = SimpleNamespace(
        terminal_fn=lambda sim: False, st={"fo": Feat(1)},
        F=SimpleNamespace(Feat=Feat), mod={"feat_copy": copy_feat})
    root = SimpleNamespace(full=None, step=10)

    def build(obs, a0):
        clock.tick(prep / 2)
        phase_calls.append("build_root")
        return root

    def gen(root_, a0):
        clock.tick(prep / 2)
        phase_calls.append("candgen")
        return ["a0", "a1"], None

    def guard(root_, labels):
        phase_calls.append("guard")
        return set()

    def rollout_fn(root_):
        def fn(s, a, ho, se):
            clock.tick(0.001)
            phase_calls.append("rollout")
            return a
        return fn

    h._build_root = build
    h._gen_candidates_raw = gen
    h._guard_screen = guard
    h._probe_sim = lambda: SimpleNamespace(
        load_observation=lambda s: None)
    h._make_rollout = rollout_fn
    return h, phase_calls


def g6_deadline_anchor_flips():
    """快照 0.6＋基线 0.2（总 0.8>0.7）：旧实现报告 0.202 并提交改
    子；加固后入口即 deadline、不做任何准备、总账 0.8。基线
    0.8>0.7：旧实现仍做 0.9 准备才回 A0；加固后零准备。"""
    clock = _Clock()
    orig = time.monotonic
    time.monotonic = clock.now
    try:
        # 反例一：快照漏计
        clock.t = 0.0
        h, calls = _fake_harness(clock, prep=0.0, begin=0.6)
        h.begin_step({"player": 0})
        clock.tick(0.2)
        d = h.decide_at_root({"step": 10}, "a0", 0.2)
        assert d is not None and d.abort_reason == "deadline", d
        assert d.changed is False and d.chosen_index == 0, d
        assert abs(d.total_elapsed_s - 0.8) < 1e-9, d.total_elapsed_s
        assert calls == ["begin_snapshot"], calls
        # 反例二：耗尽后仍做准备
        clock.t = 0.0
        h, calls = _fake_harness(clock, prep=0.9, begin=0.0)
        h.begin_step({"player": 0})
        clock.tick(0.8)
        d = h.decide_at_root({"step": 10}, "a0", 0.8)
        assert d is not None and d.abort_reason == "deadline", d
        assert calls == ["begin_snapshot"], calls
        assert abs(d.total_elapsed_s - 0.8) < 1e-9, d.total_elapsed_s
        # 旧已关反例不回退：基线 0.2＋准备 0.9 → deadline、总账 1.1
        clock.t = 0.0
        h, calls = _fake_harness(clock, prep=0.9, begin=0.0)
        h.begin_step({"player": 0})
        clock.tick(0.2)
        d = h.decide_at_root({"step": 10}, "a0", 0.2)
        assert d is not None and d.abort_reason == "deadline", d
        assert abs(d.total_elapsed_s - 1.1) < 1e-9, d.total_elapsed_s
        assert "candgen" in calls, calls
    finally:
        time.monotonic = orig


def g7_no_candidate_accounting():
    """无候选根：旧实现遥测 calls=0 不入账；加固后记一笔全调用
    耗时（abort_reason=no_candidates），返回值仍为 None。"""
    clock = _Clock()
    orig = time.monotonic
    time.monotonic = clock.now
    try:
        clock.t = 0.0
        h, _calls = _fake_harness(clock, prep=0.1)
        h._gen_candidates_raw = lambda root, a0: None
        d = h.decide_at_root({"step": 10}, "a0", 0.2)
        assert d is None, d
        snap = h.collector.snapshot()
        assert snap["calls"] == 1, snap
        assert snap["total_cost_accum_s"] > 0.2, snap
        assert snap["abort_reasons"].get("no_candidates") == 1, snap
    finally:
        time.monotonic = orig


def g8_chi_bucketed_bound():
    """C_hi 分桶：界＝桶内最大值（非均值）；新 shape 冷启动取全局
    高界；全无样本返回 None（豁免、非 0）。分桶界喂比较器：H3 桶
    0.04/次、K=2、余额 0.05 → budget_gate 整轮不启动。"""
    h = real_env.RealEnvSearch.__new__(real_env.RealEnvSearch)
    h._cand_cost_buckets = {(2, 1): [0.001, 0.002], (2, 3): [0.04]}
    assert h._est_candidate_cost_s(2, 1) == 0.002
    assert h._est_candidate_cost_s(2, 3) == 0.04
    assert h._est_candidate_cost_s(3, 3) == 0.04  # 新 shape：全局高界
    h2 = real_env.RealEnvSearch.__new__(real_env.RealEnvSearch)
    h2._cand_cost_buckets = {}
    assert h2._est_candidate_cost_s(2, 1) is None  # 全冷启动：豁免
    calls = {"n": 0}

    def rollout(state, action, hh, s):
        calls["n"] += 1
        return action

    comp = Comparator(budget_s=0.05)
    d = comp.decide(None,
                    [Candidate(0, "a0", True), Candidate(1, "a1", False)],
                    rollout, lambda a: 0.5, horizon=3, scenario_seed=0,
                    est_candidate_cost_s=h._est_candidate_cost_s(2, 3))
    assert d.abort_reason == "budget_gate" and calls["n"] == 0, d


# ===========================================================================
# R2-07 pack_bridge 加载序与绑定
# ===========================================================================

def g9_bridge_dependency_order():
    """预置 LOGC=13 的旧 feat 于 sys.modules：旧实现 np.F 捕获旧对
    象（特征 0.355）；加固后 np.F 为已验 feat（LOGC 6.5、特征
    0.710），且 clean-load 绑定恒等（search.NP is np 等）。"""
    import types
    mods = pb.load_all_sources()
    assert mods["cnet2_np"].F is mods["cnet2_feat"]
    assert mods["cnet2_search"].F is mods["cnet2_feat"]
    assert mods["cnet2_search"].NP is mods["cnet2_np"]
    assert mods["cnet2_search"].FastCarbon is mods["fastsim"].FastCarbon
    stale = types.ModuleType("cnet2_feat")
    with open(os.path.join(pb.VENDOR_DIR, "cnet2_feat.py"), "rb") as fh:
        exec(compile(fh.read(), "stale_feat", "exec"), stale.__dict__)
    stale.LOGC = 13.0
    sys.modules["cnet2_feat"] = stale
    try:
        again = pb.load_all_sources()
    finally:
        sys.modules["cnet2_feat"] = mods["cnet2_feat"]
    assert again["cnet2_np"].F is again["cnet2_feat"], "np 仍捕获旧 feat"
    assert again["cnet2_np"].F is not stale
    assert again["cnet2_feat"].LOGC == 6.5
    x = {"tok_kind": np.zeros(24, dtype=int),
         "tok_cargo": np.full(24, 100.),
         "tok_pos": np.zeros(24, dtype=int),
         "hist_cell": np.zeros((24, 8), dtype=int),
         "hist_cargo": np.zeros((24, 8)),
         "hist_move": np.zeros((24, 8), dtype=int)}
    got = float(again["cnet2_np"].Policy.tok_features(x)[0, 6])
    want = float(np.log1p(100) / 6.5)
    # tok_features 内部 float32 计算，与 float64 期望差 ~2.5e-8；
    # 判别点是合已验 feat 的 0.710 而非旧 feat（LOGC=13）的 0.355。
    assert abs(got - want) < 1e-6, (got, want)
    assert abs(got - float(np.log1p(100) / 13.0)) > 0.1, got


# ===========================================================================
# R2-06 smoke 取包
# ===========================================================================

def g10_smoke_fetch_pack_fallback():
    """缓存缺失＋checkout 在场：旧实现 ModuleNotFoundError；加固后
    经 checkout fallback 取到钉版包（整包 hash 恒验）；--pack 显式
    路径可用、错包必停。"""
    import tempfile
    import smoke_real_s1 as smoke
    real_pack = smoke.fetch_pack()  # 经既有通道取真包（缓存/在线）
    with tempfile.TemporaryDirectory() as td:
        co_pack = os.path.join(td, "code", "packs", "echo4_int8.py")
        os.makedirs(os.path.dirname(co_pack))
        with open(real_pack, "rb") as fh:
            data = fh.read()
        with open(co_pack, "wb") as fh:
            fh.write(data)
        saved_cache, saved_cands = smoke.PACK_CACHE, \
            smoke.CHECKOUT_PACK_CANDIDATES
        smoke.PACK_CACHE = os.path.join(td, "absent_pack.py")
        smoke.CHECKOUT_PACK_CANDIDATES = [co_pack]
        try:
            got = smoke.fetch_pack()
            assert os.path.normpath(got) == os.path.normpath(co_pack), got
            assert smoke.fetch_pack(pack_arg=co_pack) == co_pack
            junk = os.path.join(td, "junk.py")
            with open(junk, "wb") as fh:
                fh.write(b"not a pack")
            try:
                smoke.fetch_pack(pack_arg=junk)
            except RuntimeError:
                pass
            else:
                raise AssertionError("错包未被整包 hash 拦下")
        finally:
            smoke.PACK_CACHE, smoke.CHECKOUT_PACK_CANDIDATES = \
                saved_cache, saved_cands


TESTS = [
    ("G1 formal artifact recompute (R2-05 A8)", g1_formal_artifact_recompute),
    ("G2 synthetic positive control w/ aborts (R2-05)", g2_synthetic_positive_control_keeps_aborts),
    ("G3 synthetic NaN errors (R2-05 A5)", g3_synthetic_nan_errors),
    ("G4 synthetic missing canonical + 1 root (R2-05)", g4_synthetic_missing_canonical_one_root),
    ("G5 fixture identity params (R2-05 P3)", g5_fixture_identity_params),
    ("G6 deadline anchor flips (R2-04)", g6_deadline_anchor_flips),
    ("G7 no-candidate accounting (R2-04)", g7_no_candidate_accounting),
    ("G8 C_hi bucketed bound (R2-04)", g8_chi_bucketed_bound),
    ("G9 bridge dependency order (R2-07)", g9_bridge_dependency_order),
    ("G10 smoke fetch_pack fallback (R2-06)", g10_smoke_fetch_pack_fallback),
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
